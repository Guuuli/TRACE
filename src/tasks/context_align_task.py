import os
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from src.common import PATHS 
from src.utils.tools import  dtype_map
from .base import Tasks
from torch.utils.data.distributed import DistributedSampler
from src.utils.metrics import retrieval_precision_tensor, reciprocal_rank_tensor
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from src.utils.tools import gather_all_tensor, gather_all_tensor_with_padding, gather_all_list_strings
from src.models.mm_encoder import MultiModalEncoder
from src.utils.metrics import compute_precision_at_k, compute_mrr
from rouge_score import rouge_scorer
from multiprocessing import Pool

def compute_simiarltiy_batch(args):
    i, pred_idx, query_text, candidate_text, query_embedding, candidate_embedding = args

    # Cosine similarity (using precomputed embeddings)
    cos_sim = torch.nn.functional.cosine_similarity(query_embedding, candidate_embedding, dim=0).item()

    # ROUGE-L score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=False)
    rouge_score = scorer.score(query_text, candidate_text)["rougeL"].fmeasure

    return cos_sim, rouge_score

class ContextAligning(Tasks):
    def __init__(self, args, **kwargs):
        super().__init__(args=args, **kwargs)
        self.args = args
        self.model = MultiModalEncoder(args)
        self.num_negatives = args.num_negatives

    
    def compute_retrieval_metrics(self,
                                  query_embeddings, 
                                  candidate_embeddings,
                                  query_raw, 
                                  candidate_raw,
                                  query_labels, 
                                  candidate_labels,
                                  query_timeseries, 
                                  candidate_timeseries,
                                  retrieval_direction="text2ts",
                                  topk=10,
                                  current_epoch=0):
        """
        Utility function to compute retrieval metrics
        query_embeddings: [N, d]
        candidate_embeddings: [N, d]
        query_raw: length = N
        candidate_raw: length = N
        query_labels: [N]
        candidate_labels: [N]
        query_timeseries: [N, C, L]
        """

        similarity = torch.matmul(query_embeddings, candidate_embeddings.T)  # [N_query, N_candidate]
        indices = similarity.topk(k=topk, dim=-1).indices  # [N_query, topk]
        # ground truth: assume perfect alignment
        gt = torch.arange(similarity.shape[0], device=similarity.device)  # example: [0, 1, 2, 3, 4]
        # compute recall/precision/mrr
        mrr =  reciprocal_rank_tensor(similarity, gt)
        precision_at_1 = retrieval_precision_tensor(similarity, gt, k=1)
        precision_at_5 = retrieval_precision_tensor(similarity, gt, k=5)
        precision_at_10 = retrieval_precision_tensor(similarity, gt, k=10)
        
        print(f"Retrieval ({retrieval_direction}) results:")
        print(f"Precision@1: {precision_at_1.item():.4f}, Precision@5: {precision_at_5.item():.4f}, Precision@10: {precision_at_10.item():.4f}")
        print(f"MRR: {mrr.item():.4f}")
        
        
        if retrieval_direction == "text2ts":
            args_list = [
                (i, pred_idx.item(), query_raw[i], candidate_raw[pred_idx.item()],
                query_embeddings[i].cpu(), query_embeddings[pred_idx.item()].cpu())
                for i, pred_idx in enumerate(indices[:, 0])
            ]

            with Pool(processes=8) as pool:
                results = pool.map(compute_simiarltiy_batch, args_list)
            cosine_sims, rouge_scores = zip(*results)

            print(f"Avg ROUGE Score: {np.mean(rouge_scores):.4f}")
            print(f"Avg Cosine Similarity: {np.mean(cosine_sims):.4f}")
            
        # ts distance evaluation (only for ts2text)
        if retrieval_direction == "ts2text":
            args_list = [
                (i, pred_idx.item(), query_raw[i], candidate_raw[pred_idx.item()],
                candidate_embeddings[i].cpu(), candidate_embeddings[pred_idx.item()].cpu())
                for i, pred_idx in enumerate(indices[:, 0])
            ]

            with Pool(processes=8) as pool:
                results = pool.map(compute_simiarltiy_batch, args_list)
            cosine_sims, rouge_scores = zip(*results)

            print(f"Avg ROUGE Score: {np.mean(rouge_scores):.4f}")
            print(f"Avg Cosine Similarity: {np.mean(cosine_sims):.4f}")

        l1_distances = []
        l2_distances = []
        
        for i, preds in enumerate(indices[:, 0]):
            query_ts = query_timeseries[i]
            retrieved_ts = candidate_timeseries[preds]
            l1_dist = torch.abs(query_ts - retrieved_ts).mean().item()
            l2_dist = ((query_ts - retrieved_ts) ** 2).mean().item()
            l1_distances.append(l1_dist)
            l2_distances.append(l2_dist)    
        print(f"Avg L1 Distance: {np.mean(l1_distances):.4f}")
        print(f"Avg L2 Distance: {np.mean(l2_distances):.4f}")
        
        rouge_score = float(np.mean(rouge_scores))
        cosine_sim = float(np.mean(cosine_sims))
        l1_distance = float(np.mean(l1_distances))
        l2_distance = float(np.mean(l2_distances))
        
        precision_at_1_label = compute_precision_at_k(indices, candidate_labels, query_labels, k=1)
        precision_at_5_label = compute_precision_at_k(indices, candidate_labels, query_labels, k=5)
        
        mrr_label = compute_mrr(indices, candidate_labels, query_labels)
        
    
        print(f"Label Retrieval Accuracy (Top-1): {precision_at_1.item():.4f}, Top-5: {precision_at_5.item():.4f}")
        print(f"Label Retrieval MRR: {mrr_label.item():.4f}")
        
        log_dict = {
            "precision_at_1": precision_at_1.item(), 
            "precision_at_5": precision_at_5.item(), 
            "precision_at_10": precision_at_10.item(),
            "mrr": mrr.item(), 
            "precision_at_1_label": precision_at_1_label.item(),
            "precision_at_5_label": precision_at_5_label.item(),
            "mrr_label": mrr_label.item(),
            "rouge_score": rouge_score,
            "cosine_sim": cosine_sim,
            "MAE": l1_distance,
            "MSE": l2_distance
        }

        log_path = os.path.join(self.args.result_dir, f"{self.run_name}.txt")
        log_dir = os.path.dirname(log_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        with open(log_path, "a") as f:
            f.write(f"\nEpoch {current_epoch} - Retrieval ({retrieval_direction}) results:\n")
            for k, v in log_dict.items():
                f.write(f"{k}: {v:.4f}\n")
        
        return
    
    def evaluate_log(self, current_epoch=0):
        if self.args.distributed and isinstance(self.test_dataloader.sampler, DistributedSampler):
            self.test_dataloader.sampler.set_epoch(current_epoch)
        all_ts_embeddings = []
        all_text_embeddings = []
        all_raw_descriptions = []
        all_raw_events = []
        all_labels = []
        all_timeseries = []
        all_preds, all_valid_labels = [], []
        self.model.eval()
        with torch.no_grad():
            for batch_x in tqdm(self.test_dataloader, total=len(self.test_dataloader)):
                timeseries = batch_x.timeseries.float().to(self.device) if self.args.model_name.lower() not in ["chronos"] else batch_x.timeseries.float() #[B, C, L]
                input_mask = batch_x.input_mask.long().to(self.device)  #[B, C, L]
                labels = torch.tensor(batch_x.labels, dtype=torch.long).reshape(-1).to(self.device)  #[B]
                channel_description_emb = batch_x.channel_description_emb.to(self.device)  #[B, C, d]
                description_emb = batch_x.description_emb.to(self.device)  #[B, d]
                event_emb = batch_x.event_emb.to(self.device)  #[B, d]
                raw_description = batch_x.descriptions  # list of strings
                raw_event = batch_x.events # list of strings
                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(x_enc=timeseries, input_mask=input_mask, channel_description_emb=channel_description_emb, description_emb=description_emb, event_emb=event_emb)
                    
                del input_mask, channel_description_emb, description_emb, event_emb
                torch.cuda.empty_cache()
                preds = outputs.classification.argmax(dim=1)

                # Support -100 padding
                valid_mask = labels != -100
                all_preds.append(preds[valid_mask])
                all_valid_labels.append(labels[valid_mask])
                
                all_ts_embeddings.append(outputs.embeddings)
                all_text_embeddings.append(outputs.description_emb)
                all_raw_descriptions.extend(raw_description)
                all_raw_events.extend(raw_event)
                all_labels.append(labels)
                all_timeseries.append(timeseries)
        

            # gather across gpus
            all_ts_embeddings = gather_all_tensor(torch.cat(all_ts_embeddings, dim=0))
            all_text_embeddings = gather_all_tensor(torch.cat(all_text_embeddings, dim=0))
            all_labels = gather_all_tensor(torch.cat(all_labels, dim=0))
            all_timeseries = gather_all_tensor(torch.cat(all_timeseries, dim=0))
            all_preds = gather_all_tensor_with_padding(torch.cat(all_preds, dim=0))
            all_valid_labels = gather_all_tensor_with_padding(torch.cat(all_valid_labels, dim=0))
            all_raw_descriptions = gather_all_list_strings(all_raw_descriptions)
            all_raw_events = gather_all_list_strings(all_raw_events)
                
            # === Compute metrics === #
            all_preds_np = all_preds.cpu().numpy()
            all_valid_labels_np = all_valid_labels.cpu().numpy()
        
        
        all_ts_embeddings = nn.functional.normalize(all_ts_embeddings, dim=-1)  # [B, d]
        all_text_embeddings = nn.functional.normalize(all_text_embeddings, dim=-1)  # [B, d]
        
        if self.args.rank == 0:
            print("retrieval evaluation: text2ts")
            self.compute_retrieval_metrics(all_text_embeddings,all_ts_embeddings, all_raw_descriptions, all_raw_descriptions, all_labels, all_labels, all_timeseries, all_timeseries,retrieval_direction="text2ts", topk=10, current_epoch=current_epoch)
            
            print("retrieval evaluation: ts2text")
            self.compute_retrieval_metrics(all_ts_embeddings,all_text_embeddings, all_raw_descriptions, all_raw_descriptions, all_labels, all_labels, all_timeseries, all_timeseries, retrieval_direction="ts2text", topk=10, current_epoch=current_epoch)

            accuracy = accuracy_score(all_valid_labels_np, all_preds_np)
            precision = precision_score(all_valid_labels_np, all_preds_np, average="macro", zero_division=0)
            recall = recall_score(all_valid_labels_np, all_preds_np, average="macro", zero_division=0)
            f1 = f1_score(all_valid_labels_np, all_preds_np, average="macro", zero_division=0)    
            self.logger.log({
                "test_accuracy": accuracy,
                "test_precision": precision,
                "test_recall": recall,
                "test_f1": f1
            })
            
        return

    def train(self):
        if self.args.rank == 0:
            self.run_name = self.logger.name

        self.optimizer = self._select_optimizer()
        self.forecast_criterion = self._select_criterion()
        self.classification_criterion = nn.CrossEntropyLoss()
        self.contrastive_criterion = nn.CrossEntropyLoss()
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.args.use_amp)
        self._init_lr_scheduler(type=self.args.lr_scheduler_type)
        
        self.model.to(self.args.rank)
        self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.args.rank],find_unused_parameters=True)
        
        opt_steps = 0
        cur_epoch = 0
        self.evaluate_log(current_epoch=cur_epoch)
        while cur_epoch < self.args.max_epoch:
            print(f"Epoch {cur_epoch} of {self.args.max_epoch}")
            self.model.train()
            if self.args.distributed and isinstance(self.train_dataloader.sampler, DistributedSampler):
                self.train_dataloader.sampler.set_epoch(cur_epoch)
                
            for batch_x in tqdm(self.train_dataloader, total=len(self.train_dataloader)):
                self.optimizer.zero_grad(set_to_none=True)
                timeseries = batch_x.timeseries.float().to(self.device) if self.args.model_name.lower() not in ["chronos"] else batch_x.timeseries.float().to("cpu") #[B, C, L]
                input_mask = batch_x.input_mask.long().to(self.device)  #[B, C, L]
                labels = torch.tensor(batch_x.labels, dtype=torch.long).reshape(-1).to(self.device)
                channel_description_emb = batch_x.channel_description_emb.to(self.device) #[B, C, d]
                description_emb = batch_x.description_emb.to(self.device)  #[B, d]
                event_emb = batch_x.event_emb.to(self.device)  #[B, d]
                if not self.args.set_input_mask:
                    input_mask = torch.ones_like(input_mask)

                with torch.autocast(
                    device_type="cuda",
                    dtype=dtype_map(self.args.torch_dtype),
                    enabled=self.args.use_amp,
                ):
                    outputs = self.model(x_enc=timeseries, input_mask=input_mask, channel_description_emb=channel_description_emb, description_emb=description_emb, event_emb=event_emb)

                
                if hasattr(self.args, "model_name") and self.args.model_name.lower() in ["moment", "time-moe", "timer", "chronos"]:
                    loss = self._get_loss_tsfm(outputs, timeseries, labels, input_mask, opt_steps)
                else:
                    loss = self._get_loss(outputs, timeseries, labels, input_mask, opt_steps)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                opt_steps = opt_steps + 1

                # Adjust learning rate
                if self.args.lr_scheduler_type == "linearwarmupcosinelr":
                    self.lr_scheduler.step(cur_epoch=cur_epoch, cur_step=opt_steps)
                elif self.args.lr_scheduler_type == "onecyclelr":
                    self.lr_scheduler.step()
            
            torch.cuda.empty_cache()
            
            cur_epoch = cur_epoch + 1
            if cur_epoch % self.args.log_interval == 0:
                self.evaluate_log(current_epoch=cur_epoch)


                if self.args.rank == 0 and hasattr(self.args, "save_model") and self.args.save_model:
                    best_checkpoint_path = os.path.join(self.args.checkpoint_path, f"{self.run_name}_best_checkpoint.pt")
                    torch.save({
                        'model_state_dict': self.model.state_dict(),
                        'args': self.args
                    }, best_checkpoint_path)
                    print(f"Saved new best checkpoint to {best_checkpoint_path}")
        return self.model

    def _get_loss(self, outputs, timeseries, labels, input_mask, opt_steps):
        B, C, L = timeseries.shape
        recon_loss = self.forecast_criterion(outputs.reconstruction, timeseries)  #[B, C, L]
        observed_mask = input_mask * (1 - outputs.pretrain_mask)  #[B, C, L]
        masked_loss = observed_mask * recon_loss  #[B, C, L]
        recon_loss = masked_loss.nansum() / (observed_mask.nansum() + 1e-7)  #[B, C, L]
        labeled_mask =(labels != -100)
        if labeled_mask.any():
            classification_loss = self.classification_criterion(outputs.classification, labels)  #[B, n_classes]
        else:
            classification_loss = 0.0 * outputs.classification.sum()
        
        if self.args.hard_negative_mining == True:
            ## ✅ Sample-level contrastive loss
            ts_proj = nn.functional.normalize(outputs.embeddings, dim=-1)         # [B, d]
            text_proj = nn.functional.normalize(outputs.description_emb, dim=-1)  # [B, d]
            B = ts_proj.size(0)
            sim_matrix = torch.matmul(ts_proj, text_proj.T)  # [B, B]

            # ts → text
            labels_ts = torch.zeros(B, dtype=torch.long, device=ts_proj.device)
            mask_ts = torch.eye(B, device=ts_proj.device).bool()
            pos_logits_ts = sim_matrix[mask_ts].reshape(B, 1)
            neg_logits_ts = sim_matrix.masked_fill(mask_ts, -float('inf'))
            topk_neg_ts, _ = torch.topk(neg_logits_ts, k=self.num_negatives, dim=1)
            logits_ts = torch.cat([pos_logits_ts, topk_neg_ts], dim=1)
            loss_ts2text = self.contrastive_criterion(logits_ts / 0.07, labels_ts)

            # text → ts
            labels_text = torch.zeros(B, dtype=torch.long, device=text_proj.device)
            sim_matrix_T = sim_matrix.T
            mask_text = torch.eye(B, device=ts_proj.device).bool()
            pos_logits_text = sim_matrix_T[mask_text].reshape(B, 1)
            neg_logits_text = sim_matrix_T.masked_fill(mask_text, -float('inf'))
            topk_neg_text, _ = torch.topk(neg_logits_text, k=self.num_negatives, dim=1)
            logits_text = torch.cat([pos_logits_text, topk_neg_text], dim=1)
            loss_text2ts = self.contrastive_criterion(logits_text / 0.07, labels_text)

            loss_global = (loss_ts2text + loss_text2ts) / 2
            
            ## ✅ Channel-level contrastive loss
            ts_channel = nn.functional.normalize(outputs.channel_embeddings.reshape(B * C, -1), dim=-1)  # [B*C, d]
            text_channel = nn.functional.normalize(outputs.channel_description_emb.reshape(B * C, -1), dim=-1)  # [B*C, d]
            BC = ts_channel.size(0)
            sim_matrix = torch.matmul(ts_channel, text_channel.T)  # [B*C, B*C]

            # ts_channel → text_channel
            labels_ts = torch.zeros(BC, dtype=torch.long, device=ts_channel.device)
            mask_ts = torch.eye(BC, device=ts_channel.device).bool()
            pos_logits_ts = sim_matrix[mask_ts].reshape(BC, 1)
            neg_logits_ts = sim_matrix.masked_fill(mask_ts, -float('inf'))
            topk_neg_ts, _ = torch.topk(neg_logits_ts, k=self.num_negatives, dim=1)
            logits_ts = torch.cat([pos_logits_ts, topk_neg_ts], dim=1)
            loss_ts2text = self.contrastive_criterion(logits_ts / 0.07, labels_ts)

            # text_channel → ts_channel
            labels_text = torch.zeros(BC, dtype=torch.long, device=text_channel.device)
            sim_matrix_T = sim_matrix.T
            mask_text = torch.eye(BC, device=ts_channel.device).bool()
            pos_logits_text = sim_matrix_T[mask_text].reshape(BC, 1)
            neg_logits_text = sim_matrix_T.masked_fill(mask_text, -float('inf'))
            topk_neg_text, _ = torch.topk(neg_logits_text, k=self.num_negatives, dim=1)
            logits_text = torch.cat([pos_logits_text, topk_neg_text], dim=1)
            loss_text2ts = self.contrastive_criterion(logits_text / 0.07, labels_text)

            loss_channel = (loss_ts2text + loss_text2ts) / 2
            
            ## ✅ Event-level contrastive loss
            ts_proj = nn.functional.normalize(outputs.embeddings, dim=-1)         # [B, d]
            event_proj = nn.functional.normalize(outputs.event_emb, dim=-1)       # [B, d]
            sim_matrix = torch.matmul(ts_proj, event_proj.T)  # [B, B]

            # ts → event
            labels_ts = torch.zeros(B, dtype=torch.long, device=ts_proj.device)
            mask_ts = torch.eye(B, device=ts_proj.device).bool()
            pos_logits_ts = sim_matrix[mask_ts].reshape(B, 1)
            neg_logits_ts = sim_matrix.masked_fill(mask_ts, -float('inf'))
            topk_neg_ts, _ = torch.topk(neg_logits_ts, k=self.num_negatives, dim=1)
            logits_ts = torch.cat([pos_logits_ts, topk_neg_ts], dim=1)
            loss_ts2event = self.contrastive_criterion(logits_ts / 0.07, labels_ts)

            # event → ts
            labels_event = torch.zeros(B, dtype=torch.long, device=event_proj.device)
            sim_matrix_T = sim_matrix.T
            mask_text = torch.eye(B, device=ts_proj.device).bool()
            pos_logits_event = sim_matrix_T[mask_text].reshape(B, 1)
            neg_logits_event = sim_matrix_T.masked_fill(mask_text, -float('inf'))
            topk_neg_event, _ = torch.topk(neg_logits_event, k=self.num_negatives, dim=1)
            logits_event = torch.cat([pos_logits_event, topk_neg_event], dim=1)
            loss_event2ts = self.contrastive_criterion(logits_event / 0.07, labels_event)

            loss_event = (loss_ts2event + loss_event2ts) / 2
            
        
        else:
            ## ✅ Channel-level contrastive loss
            ts_channel = nn.functional.normalize(outputs.channel_embeddings.reshape(B * C, -1), dim=-1)  # [B*C, d]
            text_channel = nn.functional.normalize(outputs.channel_description_emb.reshape(B * C, -1), dim=-1)  # [B*C, d]
            logits_forward = torch.matmul(ts_channel, text_channel.T) / 0.07  # [B*C, B*C]
            logits_backward = logits_forward.T  # [B*C, B*C]
            labels = torch.arange(B * C, device=ts_channel.device)
            loss_fwd = self.contrastive_criterion(logits_forward, labels)
            loss_bwd = self.contrastive_criterion(logits_backward, labels)
            loss_channel = (loss_fwd + loss_bwd) / 2
            
            ## ✅ Sample-level contrastive loss
            ts = nn.functional.normalize(outputs.embeddings, dim=-1)  # [B, d]
            text = nn.functional.normalize(outputs.description_emb, dim=-1)  # [B, d]
            logits_forward = torch.matmul(ts, text.T) / 0.07  # [B, B]
            logits_backward = logits_forward.T
            labels = torch.arange(B, device=ts.device)
            loss_fwd = self.contrastive_criterion(logits_forward, labels)
            loss_bwd = self.contrastive_criterion(logits_backward, labels)
            loss_global = (loss_fwd + loss_bwd) / 2
            
            ## ✅ Event-level contrastive loss
            ts = nn.functional.normalize(outputs.cls_embedding, dim=-1)  # [B, d]
            event = nn.functional.normalize(outputs.event_emb, dim=-1)   # [B, d]
            logits_forward = torch.matmul(ts, event.T) / 0.07
            logits_backward = logits_forward.T
            labels = torch.arange(B, device=ts.device)
            loss_fwd = self.contrastive_criterion(logits_forward, labels)
            loss_bwd = self.contrastive_criterion(logits_backward, labels)

            loss_event = (loss_fwd + loss_bwd) / 2
        
        loss = recon_loss + classification_loss + loss_channel + loss_global + loss_event
        
        if self.args.rank == 0 and opt_steps % 30 == 0:
            self.logger.log(
                {
                    "train_total_loss": loss.item(),
                    "train_recon_loss": recon_loss.item(),
                    "train_classification_loss": classification_loss.item(),
                    "train_loss_channel": loss_channel.item(),
                    "train_loss_global": loss_global.item(),
                    "train_loss_event": loss_event.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        )
            if self.args.debug:
                print({
                    "train_total_loss": loss.item(),
                    "train_recon_loss": recon_loss.item(),
                    "train_classification_loss": classification_loss.item(),
                    "train_loss_channel": loss_channel.item(),
                    "train_loss_global": loss_global.item(),
                    "train_loss_event": loss_event.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
            })
                
        return loss
    

    def _get_loss_tsfm(self, outputs, timeseries, labels, input_mask, opt_steps):
        B, C, L = timeseries.shape
        labeled_mask =(labels != -100)
        if labeled_mask.any():
            classification_loss = self.classification_criterion(outputs.classification, labels)  #[B, n_classes]
        else:
            classification_loss = 0.0 * outputs.classification.sum()
        
        if self.args.hard_negative_mining == True:
            ## ✅ Sample-level contrastive loss
            ts_proj = nn.functional.normalize(outputs.embeddings, dim=-1)         # [B, d]
            text_proj = nn.functional.normalize(outputs.description_emb, dim=-1)  # [B, d]
            B = ts_proj.size(0)
            sim_matrix = torch.matmul(ts_proj, text_proj.T)  # [B, B]

            # ts → text
            labels_ts = torch.zeros(B, dtype=torch.long, device=ts_proj.device)
            mask_ts = torch.eye(B, device=ts_proj.device).bool()
            pos_logits_ts = sim_matrix[mask_ts].reshape(B, 1)
            neg_logits_ts = sim_matrix.masked_fill(mask_ts, -float('inf'))
            topk_neg_ts, _ = torch.topk(neg_logits_ts, k=self.num_negatives, dim=1)
            logits_ts = torch.cat([pos_logits_ts, topk_neg_ts], dim=1)
            loss_ts2text = self.contrastive_criterion(logits_ts / 0.07, labels_ts)

            # text → ts
            labels_text = torch.zeros(B, dtype=torch.long, device=text_proj.device)
            sim_matrix_T = sim_matrix.T
            mask_text = torch.eye(B, device=ts_proj.device).bool()
            pos_logits_text = sim_matrix_T[mask_text].reshape(B, 1)
            neg_logits_text = sim_matrix_T.masked_fill(mask_text, -float('inf'))
            topk_neg_text, _ = torch.topk(neg_logits_text, k=self.num_negatives, dim=1)
            logits_text = torch.cat([pos_logits_text, topk_neg_text], dim=1)
            loss_text2ts = self.contrastive_criterion(logits_text / 0.07, labels_text)

            loss_global = (loss_ts2text + loss_text2ts) / 2
        
            
            ## ✅ Event-level contrastive loss
            ts_proj = nn.functional.normalize(outputs.embeddings, dim=-1)         # [B, d]
            event_proj = nn.functional.normalize(outputs.event_emb, dim=-1)       # [B, d]
            sim_matrix = torch.matmul(ts_proj, event_proj.T)  # [B, B]

            # ts → event
            labels_ts = torch.zeros(B, dtype=torch.long, device=ts_proj.device)
            mask_ts = torch.eye(B, device=ts_proj.device).bool()
            pos_logits_ts = sim_matrix[mask_ts].reshape(B, 1)
            neg_logits_ts = sim_matrix.masked_fill(mask_ts, -float('inf'))
            topk_neg_ts, _ = torch.topk(neg_logits_ts, k=self.num_negatives, dim=1)
            logits_ts = torch.cat([pos_logits_ts, topk_neg_ts], dim=1)
            loss_ts2event = self.contrastive_criterion(logits_ts / 0.07, labels_ts)

            # event → ts
            labels_event = torch.zeros(B, dtype=torch.long, device=event_proj.device)
            sim_matrix_T = sim_matrix.T
            mask_text = torch.eye(B, device=ts_proj.device).bool()
            pos_logits_event = sim_matrix_T[mask_text].reshape(B, 1)
            neg_logits_event = sim_matrix_T.masked_fill(mask_text, -float('inf'))
            topk_neg_event, _ = torch.topk(neg_logits_event, k=self.num_negatives, dim=1)
            logits_event = torch.cat([pos_logits_event, topk_neg_event], dim=1)
            loss_event2ts = self.contrastive_criterion(logits_event / 0.07, labels_event)

            loss_event = (loss_ts2event + loss_event2ts) / 2
            
        
        else:
            
            ## ✅ Sample-level contrastive loss
            ts = nn.functional.normalize(outputs.embeddings, dim=-1)  # [B, d]
            text = nn.functional.normalize(outputs.description_emb, dim=-1)  # [B, d]
            logits_forward = torch.matmul(ts, text.T) / 0.07  # [B, B]
            logits_backward = logits_forward.T
            labels = torch.arange(B, device=ts.device)
            loss_fwd = self.contrastive_criterion(logits_forward, labels)
            loss_bwd = self.contrastive_criterion(logits_backward, labels)
            loss_global = (loss_fwd + loss_bwd) / 2
            
            ## ✅ Event-level contrastive loss
            ts = nn.functional.normalize(outputs.cls_embedding, dim=-1)  # [B, d]
            event = nn.functional.normalize(outputs.event_emb, dim=-1)   # [B, d]
            logits_forward = torch.matmul(ts, event.T) / 0.07
            logits_backward = logits_forward.T
            labels = torch.arange(B, device=ts.device)
            loss_fwd = self.contrastive_criterion(logits_forward, labels)
            loss_bwd = self.contrastive_criterion(logits_backward, labels)

            loss_event = (loss_fwd + loss_bwd) / 2
        
        loss = classification_loss + loss_global + loss_event
        
        if self.args.rank == 0 and opt_steps % 30 == 0:
            self.logger.log(
                {
                    "train_total_loss": loss.item(),
                    "train_classification_loss": classification_loss.item(),
                    "train_loss_global": loss_global.item(),
                    "train_loss_event": loss_event.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
        )
            if self.args.debug:
                print({
                    "train_total_loss": loss.item(),
                    "train_classification_loss": classification_loss.item(),
                    "train_loss_global": loss_global.item(),
                    "train_loss_event": loss_event.item(),
                    "learning_rate": self.optimizer.param_groups[0]["lr"],
            })
                
        return loss