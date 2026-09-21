# Gradient Inversion with Diff2Obs Defense

A repository for running gradient inversion (image reconstruction) attacks under Federated Learning (FL) frameworks using `inversefed` and the **Diff2Obs** defense strategy.

---

## 📌 Requirements

Install the required dependencies before running the code:

```bash
pip install torch torchvision pyyaml lpips numpy
```

*(Ensure `inversefed` is installed or available in your workspace root.)*

---

## 📦 Datasets & Auxiliary Data

All datasets—including auxiliary evaluation sets and synthetic generated data—are hosted on the Hugging Face Hub:

👉 **[Hugging Face Dataset Repository: `farchan07/Diff2Obs`](https://huggingface.co/datasets/farchan07/Diff2Obs)**

### Included Datasets

* **`CIFAR10`**: Benchmark CIFAR-10 evaluation data.
* **`ffhq_tiny` & `ffhq_json`**: Preprocessed FFHQ subsets and annotation metadata.
* **`imagenet_val`**: ImageNet validation set subset.
* **`synthetic`**: Synthetic data generated for defense/reconstruction experiments.


## 📁 Configuration Files

The repository includes pre-configured YAML files for different setup options:

* `configs_stylegan.yml`: Configured for StyleGAN2 experiments.
* `configs_biggan.yml`: Configured for BigGAN experiments.
* `configs_cifar.yml`: Configured for GAN-free CIFAR benchmarks.

---

## 🚀 Quick Start

Run reconstruction using any of the available configurations:

### 1. GAN-Based Reconstruction (StyleGAN2)
```bash
python run_rec_diff2obs.py --config configs_stylegan.yml 
```

### 2. GAN-Based Reconstruction (BigGAN)
```bash
python run_rec_diff2obs.py --config configs_biggan.yml 

### 3. GAN-Free Reconstruction (CIFAR)
```bash
python run_rec_diff2obs.py --config configs_cifar.yml
```

---

## ⚙️ Key Configuration Options

In your chosen `.yml` file, configure the defense settings under `defense_setting`:

```yaml
defense_method: 'diff2obs'
defense_setting:
  mode: 'Obs-I'         # Options: 'Obs-I', 'Obs-M', or 'linear'
  alpha: 0.4            # Beta distribution shape parameter
  lam: null             # Fixed lambda (if linear mode)
  mask_prob: null       # Mask probability (if Obs-M mode)
  noise: 0.01           # Standard deviation for additive noise
```

---

## 📊 Outputs & Logs

Results are automatically saved to the `output/` directory:

```text
output/
└── <exp_name>_<timestamp>/
    ├── main.log                         # Console output log
    ├── experiment_config_<timestamp>.yml # Config backup
    └── epoch_0/
        ├── <id>_gt.png                  # Ground truth image
        ├── Metrics.csv                  # Evaluation metrics (PSNR, SSIM, LPIPS)
        └── Best_Layer/
            └── <id>_gen.png             # Reconstructed image
```

---

## 📜 Citation

If you use this code or the Diff2Obs defense in your research, please cite our paper:

```bibtex
@inproceedings{raswa2026diffusion,
  title={Diffusion to Obfuscation: Time-Adaptive Synthesized Generation Against Gradient Leakage Attacks in Federated Learning},
  author={Raswa, Farchan Hakim and Lu, Chun-Shien and Wang, Jia-Ching},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2026}
}
```
