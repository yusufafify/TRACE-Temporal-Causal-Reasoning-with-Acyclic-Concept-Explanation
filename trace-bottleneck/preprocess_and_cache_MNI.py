"""
Preprocess and Cache MNI-Standardized Dataset (with Time Gap + SPD Features)

This script processes the pre-standardized dataset that has been
registered to MNI152 1mm space, N4 bias-corrected, histogram-matched, and 
Z-score normalized. The segmentation masks are also spatially registered.

CRITICAL DIFFERENCES FROM ORIGINAL PREPROCESSING:
1. NO intensity normalization (data is already Z-scored and bias-corrected)
2. Loads from preprocessed directory structure:
   - Patient-XXX_week-YYY/0000_preproc.nii.gz (T1)
   - Patient-XXX_week-YYY/0001_preproc.nii.gz (T1ce)
   - Patient-XXX_week-YYY/0002_preproc.nii.gz (T2)
   - Patient-XXX_week-YYY/0003_preproc.nii.gz (FLAIR)
   - Patient-XXX_week-YYY/seg_preproc.nii.gz (segmentation mask)
3. Reads pairings from original LUMIERE patients.json
4. Keeps spatial transforms (CropOrPad to 128x128x128)
5. Keeps Volume and SPD extraction logic

Additional fields stored per pair:
    'time_gap': float     -- (followup_week - baseline_week) / 52.0
    'spd_t': float        -- SPD at baseline timepoint (cm²)
    'spd_t1': float       -- SPD at follow-up timepoint (cm²)

Usage:
    python preprocess_and_cache_MNI.py \
        --original_lumiere_dir /path/to/original/Lumiere \
        --preprocessed_dir /path/to/preprocessed/images \
        --cache_dir ./cached_data_MNI
"""

import re
import torch
import nibabel as nib
import numpy as np
import json
import argparse
from pathlib import Path
from scipy import ndimage
from tqdm import tqdm
from datetime import datetime

# Import SPD extraction logic
from spd_extraction import calculate_2d_spd_from_mask


class MNIPreprocessor:
    """Preprocessor for MNI-standardized dataset (with time-gap + SPD features)."""

    def __init__(
        self,
        original_lumiere_dir,
        preprocessed_dir,
        cache_dir,
        spatial_size=(128, 128, 128)
    ):
        self.original_lumiere_dir = Path(original_lumiere_dir)
        self.preprocessed_dir = Path(preprocessed_dir)
        self.cache_dir = Path(cache_dir)
        self.spatial_size = spatial_size

        # Preprocessed file names
        self.modality_files = ['0000_preproc.nii.gz', '0001_preproc.nii.gz', 
                               '0002_preproc.nii.gz', '0003_preproc.nii.gz']
        self.seg_file = 'seg_preproc.nii.gz'

        # Create cache directory
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        print(f"Original LUMIERE directory: {self.original_lumiere_dir}")
        print(f"Preprocessed directory: {self.preprocessed_dir}")
        print(f"Cache directory: {self.cache_dir}")
        print(f"Spatial size: {self.spatial_size}")
        print("="*60)
        print("CRITICAL: NO intensity normalization will be applied")
        print("Data is already Z-scored and bias-corrected.")
        print("Only spatial transforms (CropOrPad) will be applied.")
        print("="*60)

    # ------------------------------------------------------------------
    # Time-gap helper
    # ------------------------------------------------------------------

    def _extract_week_number(self, case_name: str) -> float:
        """
        Extract the week number embedded in a LUMIERE case name.

        Examples:
            "Patient-001_week-000-2"  ->  0.0
            "Patient-001_week-044"    -> 44.0
            "Patient-004_week-086"    -> 86.0

        Returns:
            Week number as float, or 0.0 if no match is found.
        """
        match = re.search(r'week-(\d+)', case_name)
        if match:
            return float(match.group(1))
        return 0.0

    # ------------------------------------------------------------------
    # SPD extraction helper
    # ------------------------------------------------------------------

    def _extract_spd_from_segmentation(self, case_name: str) -> float:
        """
        Extract SPD from preprocessed segmentation mask.
        
        Args:
            case_name: Case identifier (e.g., "Patient-001_week-000-2")
            
        Returns:
            SPD value in cm², or 0.0 if segmentation not found or no enhancing tumor
        """
        # Construct path to preprocessed segmentation
        seg_file = self.preprocessed_dir / case_name / self.seg_file

        # Load segmentation and extract SPD
        if seg_file.exists():
            try:
                nii = nib.load(str(seg_file))
                seg_data = nii.get_fdata().astype(np.int32)
                
                # Extract physical voxel spacing from NIfTI header
                voxel_spacing = nii.header.get_zooms()[:3]
                
                # Calculate SPD from segmentation data
                spd_cm2 = calculate_2d_spd_from_mask(seg_data, voxel_spacing)
                
                return spd_cm2
                
            except Exception as e:
                print(f"Warning: Error extracting SPD for {case_name}: {e}")
                return 0.0

        print(f"Warning: Segmentation not found for {case_name}, SPD = 0.0")
        return 0.0

    # ------------------------------------------------------------------
    # Main preprocessing pipeline
    # ------------------------------------------------------------------

    def preprocess_all(self, patients_json='patients.json'):
        """
        Preprocess all cases in the dataset.

        Args:
            patients_json: Name of patients metadata file in original_lumiere_dir
        """
        # Load metadata from original LUMIERE directory
        patients_json_path = self.original_lumiere_dir / patients_json
        if not patients_json_path.exists():
            raise FileNotFoundError(f"Patients JSON not found: {patients_json_path}")

        with open(patients_json_path, 'r') as f:
            patients_data = json.load(f)

        # Track all temporal pairs
        all_pairs = []
        processed_cases = set()

        print("\n" + "="*60)
        print("Building temporal pairs from original patients.json...")
        print("="*60)

        # Build temporal pairs
        for patient_id, patient_info in patients_data.items():
            for case_id, case_info in patient_info.items():
                # Extract case information
                baseline_path = case_info.get('baseline', '')
                followup_path = case_info.get('followup', '')
                response = case_info.get('response', -1)

                # Extract case names from paths
                baseline_case = baseline_path.split('/')[-1]
                followup_case = followup_path.split('/')[-1]

                # Skip invalid responses
                if response not in [0, 1, 2, 3]:
                    continue

                # Verify that preprocessed directories exist
                baseline_dir = self.preprocessed_dir / baseline_case
                followup_dir = self.preprocessed_dir / followup_case

                if not baseline_dir.exists():
                    print(f"Warning: Preprocessed directory not found: {baseline_dir}")
                    continue
                if not followup_dir.exists():
                    print(f"Warning: Preprocessed directory not found: {followup_dir}")
                    continue

                all_pairs.append({
                    'patient_id': patient_id,
                    'baseline_case': baseline_case,
                    'followup_case': followup_case,
                    'rano_label': response
                })

                # Track cases to process
                processed_cases.add(baseline_case)
                processed_cases.add(followup_case)

        print(f"Found {len(all_pairs)} temporal pairs")
        print(f"Found {len(processed_cases)} unique cases to process")

        # Process all unique cases
        print("\n" + "="*60)
        print("Processing individual cases (extracting volumes, SPD, etc.)...")
        print("="*60)

        case_cache = {}
        for case_name in tqdm(sorted(processed_cases), desc="Processing cases"):
            try:
                # Load and process multimodal volume (NO intensity normalization!)
                volume = self._load_multimodal_volume(case_name)

                # Load segmentation (also compute ground truth volumes)
                seg = self._load_segmentation(case_name)
                volumes = self._compute_volumes_from_segmentation(seg)

                # Extract SPD from preprocessed segmentation
                spd_cm2 = self._extract_spd_from_segmentation(case_name)

                # Save to cache
                cache_data = {
                    'volume': volume,
                    'segmentation': seg,
                    'volumes': volumes,      # [V_CE, V_T2]
                    'spd': spd_cm2           # SPD in cm²
                }

                cache_file = self.cache_dir / f"{case_name}.pt"
                torch.save(cache_data, cache_file)

                case_cache[case_name] = cache_file

            except Exception as e:
                print(f"\nError processing {case_name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        print(f"\nSuccessfully cached {len(case_cache)} cases")

        # Now process temporal pairs
        print("\n" + "="*60)
        print("Processing temporal pairs (time gaps + SPD pairs)...")
        print("="*60)

        pair_index = []
        for idx, pair in enumerate(tqdm(all_pairs, desc="Processing pairs")):
            baseline_case = pair['baseline_case']
            followup_case = pair['followup_case']

            # Check if both cases are cached
            baseline_cache = self.cache_dir / f"{baseline_case}.pt"
            followup_cache = self.cache_dir / f"{followup_case}.pt"

            if not baseline_cache.exists() or not followup_cache.exists():
                print(f"\nSkipping pair {idx}: Missing cached data")
                continue

            try:
                # Load cached data (our own trusted files)
                baseline_data = torch.load(baseline_cache, weights_only=False)
                followup_data = torch.load(followup_cache, weights_only=False)

                # Detect new lesions
                new_lesion_flag = self._detect_new_lesion(
                    baseline_data['segmentation'],
                    followup_data['segmentation']
                )

                # Compute normalized time gap
                baseline_week = self._extract_week_number(baseline_case)
                followup_week = self._extract_week_number(followup_case)
                # Clamp to at least 1 week to avoid zero/negative gaps
                time_gap_weeks = max(1.0, followup_week - baseline_week)
                time_gap_normalized = time_gap_weeks / 52.0

                # Extract SPD values from cached case data
                spd_t = baseline_data.get('spd', 0.0)   # Baseline SPD
                spd_t1 = followup_data.get('spd', 0.0)  # Follow-up SPD

                # Create pair cache
                pair_data = {
                    'baseline_cache': str(baseline_cache),
                    'followup_cache': str(followup_cache),
                    'new_lesion_flag': new_lesion_flag,
                    'rano_label': pair['rano_label'],
                    'patient_id': pair['patient_id'],
                    'baseline_case': baseline_case,
                    'followup_case': followup_case,
                    'time_gap': time_gap_normalized,     # Time gap feature
                    'spd_t': spd_t,                      # Baseline SPD
                    'spd_t1': spd_t1                     # Follow-up SPD
                }

                # Save pair-level cache
                pair_cache_file = self.cache_dir / f"pair_{idx:04d}.pt"
                torch.save(pair_data, pair_cache_file)

                pair_index.append({
                    'pair_id': idx,
                    'patient_id': pair['patient_id'],
                    'cache_file': str(pair_cache_file),
                    'rano_label': pair['rano_label']
                })

            except Exception as e:
                print(f"\nError processing pair {idx}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Save pair index
        index_file = self.cache_dir / 'pair_index.json'
        with open(index_file, 'w') as f:
            json.dump(pair_index, f, indent=2)

        print(f"\nSuccessfully cached {len(pair_index)} temporal pairs")
        print(f"Pair index saved to: {index_file}")

        # Save cache metadata for validation
        metadata = {
            'normalize': 'none',  # NO normalization applied
            'spatial_size': list(self.spatial_size),
            'preprocessed_by_teammate': True,
            'preprocessing_details': 'MNI152 1mm, N4 bias-corrected, histogram-matched, Z-scored',
            'created_at': str(datetime.now()),
            'num_cases': len(case_cache),
            'num_pairs': len(pair_index),
            'has_time_gap': True,
            'has_spd': True
        }
        metadata_file = self.cache_dir / 'cache_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Cache metadata saved to: {metadata_file}")

        print("\n" + "="*60)
        print("PREPROCESSING COMPLETE!")
        print("="*60)
        print(f"Total cases cached: {len(case_cache)}")
        print(f"Total pairs cached: {len(pair_index)}")
        print(f"Cache directory: {self.cache_dir}")

    # ------------------------------------------------------------------
    # Volume / segmentation helpers
    # ------------------------------------------------------------------

    def _load_multimodal_volume(self, case_name):
        """
        Load all 4 modalities from preprocessed directory and stack into a single tensor.
        
        CRITICAL: NO intensity normalization is applied since data is already Z-scored!
        Only spatial resizing is performed to match model input dimensions.
        """
        case_dir = self.preprocessed_dir / case_name

        if not case_dir.exists():
            raise FileNotFoundError(f"Case directory not found: {case_dir}")

        channels = []

        for modality_file in self.modality_files:
            nii_file = case_dir / modality_file

            if not nii_file.exists():
                raise FileNotFoundError(f"Missing modality file: {nii_file}")

            nii = nib.load(str(nii_file))
            data = nii.get_fdata().astype(np.float32)
            
            # CRITICAL: NO intensity normalization!
            # Data is already Z-scored and bias-corrected by teammate
            # Only apply spatial transform
            data = self._resize_volume(data, self.spatial_size)
            channels.append(data)

        volume = np.stack(channels, axis=0)
        return torch.tensor(volume, dtype=torch.float32)

    def _load_segmentation(self, case_name):
        """Load segmentation mask from preprocessed directory and resize for model input."""
        case_dir = self.preprocessed_dir / case_name
        seg_file = case_dir / self.seg_file

        if seg_file.exists():
            try:
                nii = nib.load(str(seg_file))
                seg = nii.get_fdata().astype(np.int32)
                seg = self._resize_volume(seg, self.spatial_size, order=0)
                return seg
            except Exception as e:
                print(f"Warning: Error loading segmentation for {case_name}: {e}")

        print(f"Warning: Segmentation not found for {case_name}, using zeros")
        return np.zeros(self.spatial_size, dtype=np.int32)

    def _resize_volume(self, volume, target_size, order=1):
        """Resize volume to target size using scipy zoom."""
        factors = [t / s for t, s in zip(target_size, volume.shape)]
        return ndimage.zoom(volume, factors, order=order)

    def _compute_volumes_from_segmentation(self, seg):
        """Compute ground truth volumes from segmentation (raw cm³).

        LUMIERE HD-GLIO label convention:
        - Label 2: Enhancing Tumor (Bright on T1-CE) → V_CE (Primary RANO metric)
        - Label 1: Non-Enhancing Core / Edema (Dark on T1-CE) → V_T2/V_NE

        Note: Volumes are stored in raw cm³ in cache.
        Log transformation is applied during data loading for training.
        """
        voxel_volume_cm3 = 0.001  # 1mm³ = 0.001 cm³
        v_ce = np.sum(seg == 2) * voxel_volume_cm3
        v_t2 = np.sum(seg == 1) * voxel_volume_cm3
        return torch.tensor([v_ce, v_t2], dtype=torch.float32)

    def _detect_new_lesion(self, baseline_seg, followup_seg):
        """
        Detect if new lesions appear in follow-up.
        Returns 1.0 if new lesions detected, 0.0 otherwise.
        """
        # Extract enhancing tumor (label 2)
        baseline_ce = (baseline_seg == 2).astype(np.uint8)
        followup_ce = (followup_seg == 2).astype(np.uint8)

        # Label connected components
        baseline_labels, baseline_num = ndimage.label(baseline_ce)
        followup_labels, followup_num = ndimage.label(followup_ce)

        # If more lesions in follow-up, consider it as new lesion
        if followup_num > baseline_num:
            return 1.0

        return 0.0


def main():
    parser = argparse.ArgumentParser(
        description='Preprocess and cache MNI-standardized dataset with time-gap + SPD features'
    )
    parser.add_argument('--original_lumiere_dir', type=str, required=True,
                        help='Path to original LUMIERE directory containing patients.json')
    parser.add_argument('--preprocessed_dir', type=str, required=True,
                        help='Path to preprocessed directory with MNI-standardized images')
    parser.add_argument('--cache_dir', type=str,
                        default='../cached_data_MNI',
                        help='Directory to save cached .pt files (default: ../cached_data_MNI)')
    parser.add_argument('--spatial_size', type=int, nargs=3,
                        default=[128, 128, 128],
                        help='Target spatial size for volumes (default: 128 128 128)')

    args = parser.parse_args()

    spatial_size = tuple(args.spatial_size)

    print("="*60)
    print("MNI-Standardized Dataset Preprocessing and Caching")
    print("(Time Gap + SPD Features, NO Intensity Normalization)")
    print("="*60)
    print(f"Original LUMIERE dir: {args.original_lumiere_dir}")
    print(f"Preprocessed dir: {args.preprocessed_dir}")
    print(f"Cache directory: {args.cache_dir}")
    print(f"Spatial size: {spatial_size}")
    print("="*60)

    preprocessor = MNIPreprocessor(
        original_lumiere_dir=args.original_lumiere_dir,
        preprocessed_dir=args.preprocessed_dir,
        cache_dir=args.cache_dir,
        spatial_size=spatial_size
    )

    preprocessor.preprocess_all()


if __name__ == '__main__':
    main()
