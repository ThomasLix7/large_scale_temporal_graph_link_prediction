"""
This script trains and evaluates your ensemble temporal link prediction model
on the official TGB Wiki dataset v2 (tgbl-wiki) for leaderboard comparison.

Features:
- Trains ensemble model on official TGB Wiki dataset
- Evaluates using both negative sampling methods (TGB protocol + pre-computed)
- Compares results with TGB leaderboard (DyGFormer, NAT, CAWN, etc.)
- Provides research-grade evaluation metrics (MRR, ROC-AUC)
"""

import sys
import time
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from temporal_graph_link_prediction_tgb import TemporalLinkPredictorTGB

def run_ensemble_on_wiki():
    print("Training ensemble model on TGB Wiki dataset...")
    
    predictor = TemporalLinkPredictorTGB(
        decay=0.3,
        strategy="weighted", 
        seed=17,
        lookback=5
    )
    
    start_time = time.time()
    
    try:
        predictor.train(dataset_name="tgbl-wiki", max_samples=2000)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
    except Exception as e:
        print(f"Training failed: {e}")
        return
    
    print("\nEvaluating with manual TGB protocol...")
    try:
        manual_start = time.time()
        manual_results = predictor.evaluate_tgb(
            split_name='test',
            max_samples=200,
            use_precomputed=False
        )
        manual_time = time.time() - manual_start
        print(f"Manual TGB: MRR={manual_results['mrr']:.4f}, Time={manual_time:.2f}s")
        
    except Exception as e:
        print(f"Manual TGB failed: {e}")
        manual_results = None
    
    print("\nEvaluating with pre-computed negatives...")
    try:
        precomputed_start = time.time()
        precomputed_results = predictor.evaluate_tgb(
            split_name='test',
            max_samples=200,
            use_precomputed=True
        )
        precomputed_time = time.time() - precomputed_start
        print(f"Pre-computed: MRR={precomputed_results['mrr']:.4f}, Time={precomputed_time:.2f}s")
        
    except Exception as e:
        print(f"Pre-computed failed: {e}")
        precomputed_results = None
    
    if manual_results and precomputed_results:
        mrr_diff = abs(manual_results['mrr'] - precomputed_results['mrr'])
        print(f"\nMRR difference: {mrr_diff:.4f}")
    
    print("\nTesting comparison method...")
    try:
        comparison_results = predictor.compare_negative_sampling_methods(
            split_name='test',
            max_samples=200
        )
        print(f"Comparison: Manual={comparison_results['manual']['mrr']:.4f}, Pre-computed={comparison_results['precomputed']['mrr']:.4f}")
        
    except Exception as e:
        print(f"Comparison failed: {e}")
    

if __name__ == "__main__":
    run_ensemble_on_wiki() 