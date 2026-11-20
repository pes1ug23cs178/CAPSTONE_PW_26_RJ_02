"""References (IEEE style) and appendices."""
from pathlib import Path

FIG = Path(__file__).resolve().parent / "figs"

REFS = {
    # carried over from the Phase-2 report
    "diffseg": "R. A. Frick and M. Steinebach, “DiffSeg: Towards detecting diffusion-based inpainting attacks using multi-feature segmentation,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2024, pp. 3802–3808, doi: 10.1109/CVPRW63382.2024.00384.",
    "pudd": "A. L. Pellicer, Y. Li, and P. Angelov, “PUDD: Towards robust multi-modal prototype-based deepfake detection,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2024, pp. 3809–3817, doi: 10.1109/CVPRW63382.2024.00385.",
    "ciamarra": "A. Ciamarra, R. Caldelli, and A. Del Bimbo, “Temporal surface frame anomalies for deepfake video detection,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2024, pp. 3837–3844, doi: 10.1109/CVPRW63382.2024.00388.",
    "rama": "S. Rama, K. Gayathri, J. J. Moses Anto Devakanth, and R. Janani, “Development of deepfake detection techniques for protecting multimedia information using deep learning,” in *Proc. 3rd Int. Conf. Applied Artificial Intelligence and Computing (ICAAIC)*, 2024, pp. 786–788, doi: 10.1109/ICAAIC60222.2024.10575155.",
    "mansoor": "N. Mansoor and A. I. Iliev, “Explainable AI for DeepFake detection,” *Applied Sciences*, vol. 15, no. 2, Art. no. 725, Jan. 2025, doi: 10.3390/app15020725.",
    "lu_ssc": "L. Lu, Y. Wang, W. Zhuo, L. Zhang, G. Gao, and Y. Guo, “Deepfake detection via separable self-consistency learning,” in *Proc. IEEE Int. Conf. Image Processing (ICIP)*, 2024, pp. 3264–3270.",
    "deepsight": "T. Manju and S. Kalarani, “DeepSight: Enhancing deepfake image detection and classification through ensemble and deep learning techniques,” in *Proc. 5th Int. Conf. Image Processing and Capsule Networks (ICIPCN)*, 2024, pp. 28–35.",
    "irm": "Z. Yin, J. Wang, Y. Xiao, H. Zhao, T. Li, W. Zhou, A. Liu, and X. Liu, “Improving deepfake detection generalization by invariant risk minimization,” *IEEE Trans. Multimedia*, vol. 26, pp. 6785–6798, 2024, doi: 10.1109/TMM.2024.3355651.",
    "tsigos": "K. Tsigos, E. Apostolidis, S. Baxevanakis, S. Papadopoulos, and V. Mezaris, “Towards quantitative evaluation of explainable AI methods for deepfake detection,” in *Proc. ACM Int. Workshop on Multimedia AI against Disinformation (MAD ’24)*, 2024, doi: 10.1145/3643491.3660292.",
    # datasets and benchmarks
    "ffpp": "A. Rössler, D. Cozzolino, L. Verdoliva, C. Riess, J. Thies, and M. Nießner, “FaceForensics++: Learning to detect manipulated facial images,” in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2019, pp. 1–11.",
    "celebdf": "Y. Li, X. Yang, P. Sun, H. Qi, and S. Lyu, “Celeb-DF: A large-scale challenging dataset for DeepFake forensics,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 3207–3216.",
    "dfd": "N. Dufour and A. Gully, “Contributing data to deepfake detection research,” Google AI Blog, Sep. 24, 2019. [Online]. Available: https://research.google/blog/contributing-data-to-deepfake-detection-research/",
    "dfdc": "B. Dolhansky, J. Bitton, B. Pflaum, J. Lu, R. Howes, M. Wang, and C. Canton Ferrer, “The DeepFake Detection Challenge (DFDC) dataset,” arXiv:2006.07397, 2020.",
    "deepfakebench": "Z. Yan, Y. Zhang, X. Yuan, S. Lyu, and B. Wu, “DeepfakeBench: A comprehensive benchmark of deepfake detection,” in *Proc. Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track*, 2023.",
    "df40": "Z. Yan, T. Yao, S. Chen, Y. Zhao, X. Fu, J. Zhu, D. Luo, C. Wang, S. Ding, Y. Wu, and L. Yuan, “DF40: Toward next-generation deepfake detection,” in *Proc. Advances in Neural Information Processing Systems (NeurIPS), Datasets and Benchmarks Track*, 2024.",
    # detection methods
    "xception": "F. Chollet, “Xception: Deep learning with depthwise separable convolutions,” in *Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)*, 2017, pp. 1251–1258.",
    "facexray": "L. Li, J. Bao, T. Zhang, H. Yang, D. Chen, F. Wen, and B. Guo, “Face X-ray for more general face forgery detection,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2020, pp. 5001–5010.",
    "sbi": "K. Shiohara and T. Yamasaki, “Detecting deepfakes with self-blended images,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2022, pp. 18720–18729.",
    "univfd": "U. Ojha, Y. Li, and Y. J. Lee, “Towards universal fake image detectors that generalize across generative models,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2023, pp. 24480–24489.",
    "fa": "X. Cui, Y. Li, A. Luo, J. Zhou, and J. Dong, “Forensics Adapter: Adapting CLIP for generalizable face forgery detection,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition (CVPR)*, 2025, pp. 19207–19217.",
    "effort": "Z. Yan, J. Wang, P. Jin, K.-Y. Zhang, C. Liu, S. Chen, T. Yao, S. Ding, B. Wu, and L. Yuan, “Orthogonal subspace decomposition for generalizable AI-generated image detection,” in *Proc. 42nd Int. Conf. Machine Learning (ICML)*, PMLR, vol. 267, 2025, pp. 70268–70288.",
    "lnclip": "A. Yermakov, J. Cech, J. Matas, and M. Fritz, “Deepfake detection that generalizes across benchmarks,” arXiv:2508.06248, 2025.",
    "whyfake": "V. Vanian, A. Doumanoglou, and D. Zarpalas, “Why fake? Unveiling the semantic vocabulary of deepfake detectors,” in *Proc. IEEE/CVF Conf. Computer Vision and Pattern Recognition Workshops (CVPRW)*, 2026, arXiv:2607.07216.",
    "dfp2e": "S. Tariq, S. S. Woo, P. Singh, I. Irmalasari, S. Gupta, and D. Gupta, “From prediction to explanation: Multimodal, explainable, and interactive deepfake detection framework for non-expert users,” in *Proc. 33rd ACM Int. Conf. Multimedia (MM ’25)*, 2025, doi: 10.1145/3746027.3755786.",
    "laylens": "A. Narang, P. Gupta, L. Su, and A. Dhall, “LayLens: Improving deepfake understanding through simplified explanations,” arXiv:2507.10066, 2025.",
    # backbones, learning methods
    "swin": "Z. Liu, Y. Lin, Y. Cao, H. Hu, Y. Wei, Z. Zhang, S. Lin, and B. Guo, “Swin Transformer: Hierarchical vision transformer using shifted windows,” in *Proc. IEEE/CVF Int. Conf. Computer Vision (ICCV)*, 2021, pp. 10012–10022.",
    "clip": "A. Radford et al., “Learning transferable visual models from natural language supervision,” in *Proc. Int. Conf. Machine Learning (ICML)*, 2021, pp. 8748–8763.",
    "lora": "E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen, “LoRA: Low-rank adaptation of large language models,” in *Proc. Int. Conf. Learning Representations (ICLR)*, 2022.",
    "efficientnet": "M. Tan and Q. V. Le, “EfficientNet: Rethinking model scaling for convolutional neural networks,” in *Proc. Int. Conf. Machine Learning (ICML)*, 2019, pp. 6105–6114.",
    "ganin": "Y. Ganin, E. Ustinova, H. Ajakan, P. Germain, H. Larochelle, F. Laviolette, M. Marchand, and V. Lempitsky, “Domain-adversarial training of neural networks,” *J. Machine Learning Research*, vol. 17, no. 59, pp. 1–35, 2016.",
    "mmd": "A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Schölkopf, and A. Smola, “A kernel two-sample test,” *J. Machine Learning Research*, vol. 13, pp. 723–773, 2012.",
    "cbm": "P. W. Koh, T. Nguyen, Y. S. Tang, S. Mussmann, E. Pierson, B. Kim, and P. Liang, “Concept bottleneck models,” in *Proc. Int. Conf. Machine Learning (ICML)*, 2020, pp. 5338–5348.",
    "hinton": "G. Hinton, O. Vinyals, and J. Dean, “Distilling the knowledge in a neural network,” arXiv:1503.02531, 2015.",
    "guo": "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in *Proc. Int. Conf. Machine Learning (ICML)*, 2017, pp. 1321–1330.",
    "adamw": "I. Loshchilov and F. Hutter, “Decoupled weight decay regularization,” in *Proc. Int. Conf. Learning Representations (ICLR)*, 2019.",
    "fan": "A. Bulat and G. Tzimiropoulos, “How far are we from solving the 2D & 3D face alignment problem? (and a dataset of 230,000 3D facial landmarks),” in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 1021–1030.",
    # explanation methods
    "gradcam": "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in *Proc. IEEE Int. Conf. Computer Vision (ICCV)*, 2017, pp. 618–626.",
    "gradcampp": "A. Chattopadhay, A. Sarkar, P. Howlader, and V. N. Balasubramanian, “Grad-CAM++: Generalized gradient-based visual explanations for deep convolutional networks,” in *Proc. IEEE Winter Conf. Applications of Computer Vision (WACV)*, 2018, pp. 839–847.",
    "rollout": "S. Abnar and W. Zuidema, “Quantifying attention flow in transformers,” in *Proc. Annual Meeting of the Association for Computational Linguistics (ACL)*, 2020, pp. 4190–4197.",
    "rise": "V. Petsiuk, A. Das, and K. Saenko, “RISE: Randomized input sampling for explanation of black-box models,” in *Proc. British Machine Vision Conf. (BMVC)*, 2018.",
    "pointing": "J. Zhang, Z. Lin, J. Brandt, X. Shen, and S. Sclaroff, “Top-down neural attention by excitation backprop,” in *Proc. European Conf. Computer Vision (ECCV)*, 2016, pp. 543–559.",
    "tsne": "L. van der Maaten and G. Hinton, “Visualizing data using t-SNE,” *J. Machine Learning Research*, vol. 9, pp. 2579–2605, 2008.",
    # software
    "pytorch": "A. Paszke et al., “PyTorch: An imperative style, high-performance deep learning library,” in *Proc. Advances in Neural Information Processing Systems (NeurIPS)*, 2019, pp. 8024–8035.",
    "openclip": "G. Ilharco et al., “OpenCLIP,” Zenodo, 2021, doi: 10.5281/zenodo.5143773.",
    "timm": "R. Wightman, “PyTorch image models,” GitHub repository, 2019, doi: 10.5281/zenodo.4414861.",
    "onnxruntime": "ONNX Runtime developers, “ONNX Runtime,” 2021. [Online]. Available: https://onnxruntime.ai",
    "gradio": "A. Abid, A. Abdalla, A. Abid, D. Khan, A. Alfozan, and J. Zou, “Gradio: Hassle-free sharing and testing of ML models in the wild,” arXiv:1906.02569, 2019.",
    "faiss": "J. Johnson, M. Douze, and H. Jégou, “Billion-scale similarity search with GPUs,” *IEEE Trans. Big Data*, vol. 7, no. 3, pp. 535–547, 2021.",
}


def references():
    return [("unnumbered", "References"), ("refs", REFS)]


ACRONYMS = [
    ("AUC", "Area Under the (ROC) Curve; probability that a random fake scores higher than a random real"),
    ("CLIP", "Contrastive Language–Image Pre-training (vision-language model)"),
    ("CNN", "Convolutional Neural Network"),
    ("CRF", "Constant Rate Factor; H.264 quality setting (higher = stronger compression)"),
    ("DANN", "Domain-Adversarial Neural Network (training with gradient reversal)"),
    ("DCT", "Discrete Cosine Transform (block transform used by JPEG and H.264)"),
    ("DFD", "DeepFakeDetection dataset (Google/Jigsaw)"),
    ("EER", "Equal Error Rate"),
    ("EMA", "Exponential Moving Average of model weights"),
    ("FF++", "FaceForensics++ dataset"),
    ("FFT", "Fast Fourier Transform"),
    ("GFLOPs", "Billions of floating-point operations"),
    ("GRACE", "Generalisable, Robust And Concept-Explainable detection (this project's system)"),
    ("GRL", "Gradient Reversal Layer"),
    ("HLD / LLD", "High-Level Design / Low-Level Design"),
    ("IoU", "Intersection over Union"),
    ("KD", "Knowledge Distillation"),
    ("LoRA", "Low-Rank Adaptation (parameter-efficient fine-tuning)"),
    ("MMD", "Maximum Mean Discrepancy"),
    ("ONNX", "Open Neural Network Exchange (portable model format)"),
    ("PRS / SRS", "Project / Software Requirements Specification"),
    ("ROC", "Receiver Operating Characteristic"),
    ("SBI", "Self-Blended Images"),
    ("SFD", "Single-shot scale-invariant Face Detector"),
    ("t-SNE", "t-distributed Stochastic Neighbour Embedding"),
    ("ViT", "Vision Transformer"),
    ("XAI", "Explainable Artificial Intelligence"),
]

DEFINITIONS = [
    ("Deepfake", "An image or video in which a face has been swapped, re-animated or edited by a generative model."),
    ("Cross-dataset evaluation", "Testing on a dataset whose videos, actors and manipulation methods were not used for training."),
    ("Data leakage", "Information from validation or test data influencing training or model selection, which inflates reported accuracy."),
    ("Concept bottleneck", "A model layer that predicts named concepts, from which the final decision is computed."),
    ("Concept share", "The fraction of the variance of the decision logit explained by the concept part of the decision."),
    ("Self-blended image", "A pseudo-fake made by blending a face with a transformed copy of itself."),
    ("Teacher / student", "A large accurate model and a small fast model trained to imitate it."),
    ("Calibration", "Adjusting predicted probabilities so that they match observed frequencies (temperature scaling)."),
]


def appendix_a():
    return [
        ("unnumbered", "Appendix A: Definitions, Acronyms and Abbreviations", "A"),
        ("h2", "Definitions"),
        ("tab", "defs", "Definitions of key terms.", ["Term", "Definition"], [[a, b] for a, b in DEFINITIONS], [4.4, 12.6]),
        ("h2", "Acronyms and Abbreviations"),
        ("tab", "acr", "Acronyms and abbreviations.", ["Acronym", "Meaning"], [[a, b] for a, b in ACRONYMS], [3.4, 13.6]),
    ]


def appendix_b():
    return [
        ("unnumbered", "Appendix B: User Manual", "B"),
        ("h2", "Installation"),
        ("num", [
            "Install Python 3.12 and, for the Careful check, an NVIDIA driver with CUDA 12.8 support.",
            "In the project folder, install the dependencies: `pip install -r requirements.txt`.",
            "Make sure the release folder contains the teacher checkpoint, the student ONNX model and the calibration files "
            "(created by `python scripts/make_release.py`).",
        ]),
        ("h2", "Starting the Application"),
        ("p", "On Windows, set `TORCHDYNAMO_DISABLE=1` (needed by the face-alignment library) and run `python app/app.py`. "
              "Open http://127.0.0.1:7860 in a web browser. The page in Figure {fig:empty} appears."),
        ("fig", "empty", FIG / "app_empty.png", "Start page: upload area, type of check and the Check button.", 14.0),
        ("h2", "Checking a Photo or Video"),
        ("num", [
            "Drag a photo (JPG, PNG, WEBP, BMP) or a video (MP4, MOV, AVI, MKV, WEBM; up to 200 MB) into the upload area, "
            "or click it to choose a file.",
            "Choose the type of check. **Careful check (recommended)** uses the most accurate model and needs a GPU; "
            "**Quick check** runs on any computer but makes more mistakes.",
            "Click **Check this file**. A 10-second video takes about 3–5 seconds.",
            "Read the verdict card: *Likely fake*, *Can't tell* or *Looks real*, with the 'How sure are we?' scale and "
            "the number of face views used.",
            "Read the clue cards under 'Why we think so'. The orange glow shows where on the face each clue was found.",
            "Use **Save this result** to download a report, or **Check another file** to start again.",
        ]),
        ("h2", "Technical Details for Experts"),
        ("p", "Open 'Show technical details (for experts)' to see the calibrated probability and the model used, the "
              "score of every face over time, the concept maps of the most suspicious faces (choose the concept and the "
              "overlay opacity), the contribution of each concept, a switch-off test that re-scores the video with chosen "
              "concepts removed, the three most similar training faces, and the full HTML report with the file's "
              "SHA-256 hash."),
        ("h2", "Troubleshooting"),
        ("tab", "trouble", "Common messages and what to do.",
         ["Message or problem", "What to do"],
         [["'Choose a file first'", "Upload a file before pressing Check."],
          ["'We can't read this file type'", "Convert the file to MP4 (video) or JPG/PNG (photo)."],
          ["'This file is too large'", "Upload a file under 200 MB or a shorter clip."],
          ["'We couldn't find a face'", "Use a clip in which a face is clearly visible and facing the camera."],
          ["Verdict is 'Can't tell' because faces are turned", "Use a clip with a more frontal view of the face."],
          ["Page does not open", "Check that app/app.py is running and that port 7860 is free."]],
         [6.4, 10.6]),
        ("h2", "Using the API"),
        ("p", "The same analysis is available programmatically, for example with the gradio_client package: "
              "`Client(\"http://127.0.0.1:7860/\").predict(handle_file(path), \"Careful check (recommended)\", "
              "\"Blend strength\", 0.45, api_name=\"/check\")`. The endpoints are listed in Table {tab:api}."),
    ]
