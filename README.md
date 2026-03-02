# Awesome Computer Vision: From Foundations to Research
*A structured learning reference: image fundamentals → classical methods → deep learning → generative artificial intelligence in CV.*

[![Awesome](https://awesome.re/badge.svg)](https://github.com/mawady/awesome-cv)
![GitHub last commit](https://img.shields.io/github/last-commit/mawady/awesome-cv)
![GitHub stars](https://img.shields.io/github/stars/mawady/awesome-cv?style=social)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

---

## Why this list exists

There are dozens of "awesome computer vision" repositories on GitHub. Most are
encyclopedic with thousands of links arranged by topic, with no guidance on where
to start, what order to read things in, or why one resource matters more than
another. They are useful as archives. They are less useful as learning tools.

This list is built around a different idea: **curation over comprehensiveness**.

Every entry is here because it genuinely helps someone understand computer
vision more deeply — not simply because it exists. Resources are organised to
reflect how the field is actually learned: from image fundamentals and classical
methods, through deep learning, to the transformer-era models that define current
research.

### What makes this different

| | This list | Most other CV lists |
|---|---|---|
| **Paper context** | ✅ Why each paper matters, in sequence | ❌ Flat citation lists |
| **Evaluation metrics** | ✅ Full breakdown per task | ❌ Rarely covered |
| **Actively maintained** | ✅ Updated with recent work | ⚠️ Many are abandoned |
| **Conference & journal tiers** | ✅ CORE-ranked, explained | ❌ Usually just a list |
| **Multi-language libraries** | ✅ Python, Rust, MATLAB | ❌ Python only |

### Who this is for

- **Students** starting a CV module or thesis who want a clear first step
- **Engineers** moving into CV who need to fill gaps systematically  
- **Researchers** wanting a compact reference for venues, metrics, and landmark papers
- **Educators** looking for a syllabus scaffold they can point students to

> 💡 **New to the field?** Start at [Courses](#courses) or [Reference Books](#reference-books).  
> 🔬 **Already in research?** Jump to [Popular Articles](#popular-articles) or [Repos](#repos).

---

## Contents

>
> * **[Python Libraries](#python-libraries)**
> * **[Rust Libraries](#rust-libraries)**
> * **[MATLAB/Octave Libraries](#matlab-libraries)**
> * **[Evaluation Metrics](#evaluation-metrics)**
> * **[Conferences](#conferences)**
> * **[Journals](#journals)**
> * **[Summer Schools](#summer-schools)**
> * **[Popular Articles](#popular-articles)**
> * **[Reference Books](#reference-books)**
> * **[Courses](#courses)**
> * **[Repos](#repos)**
> * **[Dataset Collections](#dataset-collections)**
> * **[Annotation Tools](#annotation-tools)**
> * **[YouTube Channels](#youtube-channels)**
> * **[Mailing Lists](#mailing-lists)**
>

---

## Python Libraries

> Status: ✅ active (updated within 2 years) · ⚠️ legacy (unmaintained but historically useful) · 🗄️ archived (officially abandoned)

| Library | Description | Status |
| --- | --- | --- |
| [OpenCV](https://github.com/opencv/opencv) | Open Source Computer Vision Library | ✅ active |
| [Pillow](https://github.com/python-pillow/Pillow) | The friendly PIL fork (Python Imaging Library) | ✅ active |
| [scikit-image](https://github.com/scikit-image/scikit-image) | Collection of algorithms for image processing | ✅ active |
| [SciPy](https://github.com/scipy/scipy) | Open-source software for mathematics, science, and engineering | ✅ active |
| [mmcv](https://github.com/open-mmlab/mmcv) | OpenMMLab foundational library for computer vision research | ✅ active |
| [imutils](https://github.com/PyImageSearch/imutils) | Convenience functions for basic image processing operations | ✅ active |
| [kornia](https://github.com/kornia/kornia) | Open source differentiable computer vision library for PyTorch | ✅ active |
| [pgmagick](https://github.com/hhatto/pgmagick) | Python wrapper for GraphicsMagick/ImageMagick | ⚠️ legacy |
| [Mahotas](https://github.com/luispedro/mahotas) | Fast computer vision algorithms in Python | ⚠️ legacy |
| [SimpleCV](https://github.com/sightmachine/SimpleCV) | Open Source Framework for Machine Vision | 🗄️ archived |

---

## Rust Libraries

> Status: ✅ active (updated within 2 years) · ⚠️ legacy (unmaintained but historically useful) · 🗄️ archived (officially abandoned)

| Library | Description | Status |
| --- | --- | --- |
| [OpenCV-Rust](https://github.com/twistedfall/opencv-rust) | Rust bindings for OpenCV 3.4, 4.x, and 5.x | ✅ active |
| [Image](https://github.com/image-rs/image) | Encoding and decoding images in Rust | ✅ active |
| [ImageProc](https://github.com/image-rs/imageproc) | Image processing operations built on the image crate | ✅ active |
| [Photon](https://github.com/silvia-odwyer/photon) | Rust/WebAssembly image processing library | ⚠️ legacy |

---

## MATLAB Libraries

> Status: ✅ active (updated within 2 years) · ⚠️ legacy (unmaintained but historically useful) · 🗄️ archived (officially abandoned)

| Library | Description | Status |
| --- | --- | --- |
| [MLV](https://github.com/bwlabToronto/MLV_toolbox) | Mid-level Vision Toolbox, BWLab, University of Toronto | ✅ active |
| [PMT](https://pdollar.github.io/toolbox/) | Piotr's Computer Vision MATLAB Toolbox, P. Dollar | ⚠️ legacy |
| [matlabfns](https://www.peterkovesi.com/matlabfns/) | MATLAB and Octave functions for computer vision and image processing, P. Kovesi, University of Western Australia | ⚠️ legacy |
| [VLFeat](https://www.vlfeat.org/index.html) | Open source library of popular CV algorithms (SIFT, VLAD, Fisher Vectors, SLIC), A. Vedaldi and B. Fulkerson | ⚠️ legacy |
| [ElencoCode](https://www.dropbox.com/s/bguw035yrqz0pwp/ElencoCode.docx?dl=0) | Loris Nanni's CV functions, University of Padova | ⚠️ legacy |

---

## Evaluation Metrics

* Performance - Classification
  * Confusion Matrix: TP, FP, TN, and FN for each class
  * For class-balanced datasets:
    * Accuracy: (TP+TN) / (TP+FP+TN+FN)
    * ROC curve: TPR vs FPR · summarised by AUROC (higher is better)
  * For class-imbalanced datasets:
    * Precision (P): TP / (TP+FP)
    * Recall (R): TP / (TP+FN)
    * F1-Score: 2·P·R / (P+R)
    * Balanced Accuracy: (TPR+TNR) / 2
    * Weighted-Averaged Precision, Recall, and F1-Score
    * PR curve: Precision vs Recall · summarised by AUPRC (higher is better, more informative than AUROC on imbalanced data)
  * For multi-label classification:
    * Macro / Micro / Weighted averaging of above metrics
    * Hamming Loss: fraction of labels incorrectly predicted

* Performance - Detection
  * Intersection over Union (IoU): area of overlap / area of union between predicted and ground-truth box
  * Average Precision (AP): area under the Precision-Recall curve for a single class
  * mAP: mean AP averaged over all classes
  * mAP@0.5: IoU threshold of 0.5 (PASCAL VOC standard)
  * mAP@0.5:0.95: mean over IoU thresholds 0.5 to 0.95 in steps of 0.05 (COCO standard, harder and preferred)
  * AR@k: Average Recall at k proposals per image
  * False Positives Per Image (FPPI): used in pedestrian detection benchmarks (e.g. Caltech)
  * Log-Average Miss Rate (LAMR): standard metric for pedestrian detection, computed on FPPI vs Miss Rate curve

* Performance - Segmentation
  * Intersection over Union (IoU) / Jaccard Index: TP / (TP+FP+FN) per class
  * mean IoU (mIoU): IoU averaged over all classes · primary metric for semantic segmentation benchmarks (Cityscapes, ADE20K)
  * Dice Coefficient / F1-Score: 2·TP / (2·TP+FP+FN) · standard for medical image segmentation
  * Mean Pixel Accuracy (mPA): fraction of pixels correctly classified per class, then averaged
  * Panoptic Quality (PQ): PQ = SQ · RQ · unified metric for panoptic segmentation (COCO Panoptic)
  * Boundary IoU (BIoU): IoU computed only near object boundaries · penalises coarse masks
  * Hausdorff Distance (HD): maximum surface distance between predicted and ground-truth masks · common in medical imaging
  * HD95: 95th-percentile Hausdorff Distance · more robust to outliers than HD

* Performance - Tracking
  * Multiple Object Tracking Accuracy (MOTA): combines false positives, false negatives, and identity switches
  * Multiple Object Tracking Precision (MOTP): average localisation precision of matched detections
  * ID F1-Score (IDF1): ratio of correctly identified detections over average of ground-truth and computed detections · better reflects long-term identity consistency than MOTA
  * HOTA (Higher Order Tracking Accuracy): geometric mean of detection and association accuracy · increasingly preferred over MOTA/MOTP as a single summary metric
  * Identity Switches (IDSW): number of times a tracked object changes its assigned ID
  * Mostly Tracked (MT) / Mostly Lost (ML): fraction of ground-truth trajectories tracked for more than 80% / less than 20% of their lifespan

* Performance - Perceptual Quality (Super-resolution, Denoising, Enhancement)
  * Reference-based (require a clean ground-truth image):
    * Peak Signal-to-Noise Ratio (PSNR): 10·log10(MAX² / MSE) · in dB, higher is better · fast to compute but weakly correlated with human perception
    * Structural Similarity Index (SSIM): measures luminance, contrast, and structure jointly · range [0,1], higher is better
    * Multi-Scale SSIM (MS-SSIM): SSIM computed at multiple resolutions · more robust to viewing distance
    * Learned Perceptual Image Patch Similarity (LPIPS): deep feature distance · strongly correlated with human judgement · lower is better
    * Visual Information Fidelity (VIF): mutual information between reference and distorted image features
  * No-reference (blind, no ground-truth required):
    * Natural Image Quality Evaluator (NIQE): lower is better · measures deviation from natural scene statistics
    * BRISQUE: lower is better · spatial natural scene statistics
    * Gradient Magnitude Similarity Deviation (GMSD): fast, gradient-based · lower is better

* Performance - Generation (GANs, Diffusion Models)
  * Fréchet Inception Distance (FID): distance between Inception feature distributions of real and generated images · lower is better · primary benchmark metric
  * Inception Score (IS): measures quality and diversity jointly using classifier confidence and entropy · higher is better · less reliable than FID on its own
  * Kernel Inception Distance (KID): like FID but uses MMD instead of Gaussian assumption · unbiased with small sample sizes · lower is better
  * Perceptual Path Length (PPL): smoothness of the latent space · used for GANs · lower is better
  * CLIP Score: cosine similarity between CLIP embeddings of generated image and text prompt · used for text-to-image evaluation · higher is better
  * Human Evaluation: side-by-side preference studies remain the gold standard for generative quality

* Performance - Depth Estimation
  * Absolute Relative Error (AbsRel): mean( |d - d*| / d* ) · lower is better
  * Squared Relative Error (SqRel): mean( |d - d*|² / d* )
  * Root Mean Squared Error (RMSE) and RMSE log
  * Threshold Accuracy (δ < 1.25, 1.25², 1.25³): fraction of pixels where max(d/d*, d*/d) < threshold · higher is better

* Performance - Pose Estimation
  * Percentage of Correct Keypoints (PCK): keypoint within α · torso diameter of ground truth · PCK@0.2 is standard
  * Object Keypoint Similarity (OKS): analogous to IoU for keypoints · accounts for keypoint visibility and scale · used by COCO
  * Mean Per Joint Position Error (MPJPE): average Euclidean distance between predicted and ground-truth 3D joints · in mm

* Computation
  * Latency: end-to-end inference time per image (ms) · report hardware, batch size, and input resolution
  * Throughput: Frames Per Second (FPS) · report the same context as latency
  * Parameters (M): total trainable parameter count · proxy for memory footprint
  * FLOPs / MACs: floating-point operations or multiply-accumulate operations per forward pass · hardware-independent complexity measure
  * Model Size (MB): weight file size on disk
  * GPU Memory (VRAM, GB): peak memory during inference · critical for deployment constraints

---

## Conferences

> Ranks follow [CORE Conference Ranking](http://portal.core.edu.au/conf-ranks/). Acceptance rates are approximate, based on recent editions. Note: in CV and ML, conference prestige often exceeds journal prestige, unlike in most other fields.

* CORE Rank A\*
  * [CVPR](https://cvpr.thecvf.com): Conference on Computer Vision and Pattern Recognition (IEEE) · ~22% acceptance · the highest-volume top-tier CV venue [[dblp](https://dblp.org/streams/conf/cvpr)]
  * [ICCV](https://iccv.thecvf.com): International Conference on Computer Vision (IEEE) · ~26% acceptance · held in odd years only [[dblp](https://dblp.org/streams/conf/iccv)]
  * [NeurIPS](https://neurips.cc): Conference on Neural Information Processing Systems · ~26% acceptance · primary venue for ML theory and deep learning [[dblp](https://dblp.org/streams/conf/nips)]
  * [ICML](https://icml.cc): International Conference on Machine Learning · ~28% acceptance · top ML venue with growing CV presence [[dblp](https://dblp.org/streams/conf/icml)]
  * [ICLR](https://iclr.cc): International Conference on Learning Representations · ~32% acceptance · open-review format; major venue for deep learning and VLMs [[dblp](https://dblp.org/streams/conf/iclr)]
  * [ECCV](https://eccv.ecva.net): European Conference on Computer Vision (Springer) · ~28% acceptance · held in even years only [[dblp](https://dblp.org/streams/conf/eccv)]
  * [AAAI](https://aaai.org/conference/aaai): AAAI Conference on Artificial Intelligence · ~20% acceptance · broad AI scope with strong CV track [[dblp](https://dblp.org/streams/conf/aaai)]
  * [ACMMM](https://acmmm.org): ACM International Conference on Multimedia (ACM) [[dblp](https://dblp.org/streams/conf/mm)]
  * [ICRA](https://ieee-icra.org): International Conference on Robotics and Automation (IEEE) [[dblp](https://dblp.org/streams/conf/icra)]

* CORE Rank A
  * [MICCAI](https://miccai.org): Conference on Medical Image Computing and Computer Assisted Intervention (Springer) · ~30% acceptance · premier venue for medical imaging [[dblp](https://dblp.org/streams/conf/miccai)]
  * [WACV](https://wacv.thecvf.com): Winter Conference on Applications of Computer Vision (IEEE) · ~29% acceptance · practical and applied CV; growing rapidly [[dblp](https://dblp.org/streams/conf/wacv)]
  * [IROS](https://ieee-iros.org): International Conference on Intelligent Robots and Systems (IEEE) · covers CV for robotics and perception [[dblp](https://dblp.org/streams/conf/iros)]
  * [ISBI](https://biomedicalimaging.org): IEEE International Symposium on Biomedical Imaging (IEEE) [[dblp](https://dblp.org/streams/conf/isbi)]
  * [BMVC](https://www.bmva.org/bmvc): British Machine Vision Conference (BMVA) [[dblp](https://dblp.org/streams/conf/bmvc)]

* CORE Rank B
  * [ICPR](http://www.wikicfp.com/cfp/program?id=1448): International Conference on Pattern Recognition (IEEE) [[dblp](https://dblp.org/streams/conf/icpr)]
  * [ACCV](http://www.wikicfp.com/cfp/program?id=22): Asian Conference on Computer Vision (Springer) [[dblp](https://dblp.org/streams/conf/accv)]
  * [ICASSP](https://ieeeicassp.org): International Conference on Acoustics, Speech, and Signal Processing (IEEE) [[dblp](https://dblp.org/streams/conf/icassp)]
  * [ICIP](http://www.wikicfp.com/cfp/program?id=1390): International Conference on Image Processing (IEEE) [[dblp](https://dblp.org/streams/conf/icip)]
  * [VISAPP](https://visapp.scitevents.org): International Conference on Vision Theory and Applications (SCITEPRESS) [[dblp](https://dblp.org/streams/conf/visapp)]
  * [ACIVS](http://www.wikicfp.com/cfp/program?id=34): Conference on Advanced Concepts for Intelligent Vision Systems (Springer) [[dblp](https://dblp.org/streams/conf/acivs)]
  * [EUSIPCO](https://eurasip.org/eusipco-conferences/): European Signal Processing Conference (EURASIP/IEEE) [[dblp](https://dblp.org/streams/conf/eusipco)]

* CORE Rank C
  * [VCIP](http://www.wikicfp.com/cfp/program?id=2926): International Conference on Visual Communications and Image Processing (IEEE) [[dblp](https://dblp.org/streams/conf/vcip)]
  * [CAIP](http://www.wikicfp.com/cfp/program?id=346): International Conference on Computer Analysis of Images and Patterns (Springer) [[dblp](https://dblp.org/streams/conf/caip)]
  * [ICISP](http://www.wikicfp.com/cfp/program?id=1399): International Conference on Image and Signal Processing (Springer) [[dblp](https://dblp.org/streams/conf/icisp)]
  * [ICIAR](http://www.wikicfp.com/cfp/program? id=1381): International Conference on Image Analysis and Recognition (Springer) [[dblp](https://dblp.org/streams/conf/iciar)]
  * [ICVS](http://www.wikicfp.com/cfp/program?id=1501): International Conference on Computer Vision Systems (Springer) [[dblp](https://dblp.org/streams/conf/icvs)]

* Unranked but notable
  * [MIUA](https://www.bmva.org/miua): Medical Image Understanding and Analysis (BMVA) · UK-focused medical imaging [[dblp](https://dblp.org/streams/conf/miua)]
  * [EUVIP](https://eurasip.org/workshops/): European Workshop on Visual Information Processing (IEEE/EURASIP) [[dblp](https://dblp.org/streams/conf/euvip)]
  * [CIC](https://www.imaging.org/site/IST/Conferences/Color_and_Imaging): Color and Imaging Conference (IS&T) [[dblp](https://dblp.org/streams/conf/imaging)]
  * [CVCS](https://www.cvcs.no): Colour and Visual Computing Symposium [[dblp](https://dblp.org/streams/conf/cvcs)]
  * DSP: International Conference on Digital Signal Processing (IEEE) [[dblp](https://dblp.org/streams/conf/icdsp)]

---

## Journals

> Rankings use the [SCImago Journal Rank (SJR)](https://www.scimagojr.com/journalrank.php) indicator. SJR is a size-independent prestige metric: it weights citations by the influence of the citing journal, not just their count. Quartiles (Q1 to Q4) place each journal within its subject category; Q1 is the top 25%. In computer vision and machine learning, top conferences (CVPR, ICCV, ECCV) often carry more prestige than journals; many researchers publish conference papers first and submit extended versions to journals later.

* Core CV and ML Journals
  * [IEEE TPAMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34): Transactions on Pattern Analysis and Machine Intelligence · Q1 · the highest-prestige journal in CV/ML; publishes foundational and survey work [[dblp](https://dblp.org/streams/journals/pami)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=24254&tip=sid)]
  * [Elsevier MedIA](https://www.journals.elsevier.com/medical-image-analysis): Medical Image Analysis · Q1 · leading venue in medical imaging [[dblp](https://dblp.org/streams/journals/mia)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=17271&tip=sid)]
  * [IEEE TIP](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=83): Transactions on Image Processing · Q1 · image processing, analysis, and low-level vision [[dblp](https://dblp.org/streams/journals/tip)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=25534&tip=sid)]
  * [IEEE TMI](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=42): Transactions on Medical Imaging · Q1 · premier journal for medical image analysis [[dblp](https://dblp.org/streams/journals/tmi)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=16733&tip=sid)]
  * [Elsevier PR](https://www.journals.elsevier.com/pattern-recognition): Pattern Recognition · Q1 · broad scope; high volume [[dblp](https://dblp.org/streams/journals/pr)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=24823&tip=sid)]
  * [IJCV](https://www.springer.com/journal/11263): International Journal of Computer Vision (Springer) · Q1 · primary venue for long-form CV research [[dblp](https://dblp.org/streams/journals/ijcv)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=72242&tip=sid)]
  * [IEEE TCSVT](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=76): Transactions on Circuits and Systems for Video Technology · Q1 · video understanding, compression, and streaming [[dblp](https://dblp.org/streams/journals/tcsv)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=26027&tip=sid)]
  * [IEEE TVCG](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=2945): Transactions on Visualization and Computer Graphics · Q1 · covers rendering, visual analytics, and 3D vision [[dblp](https://dblp.org/streams/journals/tvcg)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=25535&tip=sid)]
  * [Elsevier CVIU](https://www.journals.elsevier.com/computer-vision-and-image-understanding): Computer Vision and Image Understanding · Q1 [[dblp](https://dblp.org/streams/journals/cviu)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=24161&tip=sid)]

* Robotics and Automation
  * [IEEE RAL](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=7083369): Robotics and Automation Letters · Q1 · fast-track letters; papers often presented at ICRA or IROS [[dblp](https://dblp.org/streams/journals/ral)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=21100900379&tip=sid)]

* Applied and Interdisciplinary
  * [Elsevier ESWA](https://www.journals.elsevier.com/expert-systems-with-applications): Expert Systems with Applications · Q1 · broad applied scope; high volume [[dblp](https://dblp.org/streams/journals/eswa)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=24201&tip=sid)]
  * [Elsevier Neurocomputing](https://www.journals.elsevier.com/neurocomputing) · Q1 [[dblp](https://dblp.org/streams/journals/ijon)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=24807&tip=sid)]
  * [Springer NCA](https://www.springer.com/journal/521): Neural Computing and Applications · Q1 [[dblp](https://dblp.org/streams/journals/nca)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=24800&tip=sid)]
  * [Elsevier CMIG](https://www.journals.elsevier.com/computerized-medical-imaging-and-graphics): Computerized Medical Imaging and Graphics · Q1 [[dblp](https://dblp.org/streams/journals/cmig)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=23607&tip=sid)]
  * [Elsevier CMPB](https://www.journals.elsevier.com/computer-methods-and-programs-in-biomedicine): Computer Methods and Programs in Biomedicine · Q1 [[dblp](https://dblp.org/streams/journals/cmpb)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=23604&tip=sid)]
  * [Elsevier CBM](https://www.journals.elsevier.com/computers-in-biology-and-medicine): Computers in Biology and Medicine · Q1 [[dblp](https://dblp.org/streams/journals/cbm)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=17957&tip=sid)]

* Specialist and Lower-Tier
  * [Elsevier PRL](https://www.journals.elsevier.com/pattern-recognition-letters): Pattern Recognition Letters · Q1 · shorter-format work [[dblp](https://dblp.org/streams/journals/prl)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=24825&tip=sid)]
  * [Elsevier IVC](https://www.journals.elsevier.com/image-and-vision-computing): Image and Vision Computing · Q1 [[dblp](https://dblp.org/streams/journals/ivc)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=25549&tip=sid)]
  * [Elsevier JVCIR](https://www.journals.elsevier.com/journal-of-visual-communication-and-image-representation): Journal of Visual Communication and Image Representation · Q2 [[dblp](https://dblp.org/streams/journals/jvcir)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=25592&tip=sid)]
  * [Springer JMIV](https://www.springer.com/journal/10851): Journal of Mathematical Imaging and Vision · Q2 · mathematical foundations of imaging [[dblp](https://dblp.org/streams/journals/jmiv)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=28501&tip=sid)]
  * [SPIE JEI](https://www.spiedigitallibrary.org/journals/journal-of-electronic-imaging): Journal of Electronic Imaging · Q3 [[dblp](https://dblp.org/streams/journals/jei)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=25978&tip=sid)]
  * [IET Image Processing](https://ietresearch.onlinelibrary.wiley.com/journal/17519667) · Q2 [[dblp](https://dblp.org/streams/journals/iet-ipr)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=5400152646&tip=sid)]
  * [Springer PAA](https://www.springer.com/journal/10044): Pattern Analysis and Applications · Q2 [[dblp](https://dblp.org/streams/journals/paa)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=24822&tip=sid)]
  * [Springer MVA](https://www.springer.com/journal/138): Machine Vision and Applications · Q2 [[dblp](https://dblp.org/streams/journals/mva)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=12984&tip=sid)]
  * [IET Computer Vision](https://ietresearch.onlinelibrary.wiley.com/journal/17519640) · Q2 [[dblp](https://dblp.org/streams/journals/iet-cvi)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=7000153231&tip=sid)]

* Open Access
  * [IEEE Access](https://ieeeaccess.ieee.org) · Q1 · broad scope; fast publication; lower selectivity than the IEEE transactions [[dblp](https://dblp.org/streams/journals/access)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=21100374601&tip=sid)]
  * [MDPI Journal of Imaging](https://www.mdpi.com/journal/jimaging) · Q2 · fully open access; no subscription required [[dblp](https://dblp.org/streams/journals/jimaging)] [[scimago](https://www.scimagojr.com/journalsearch.php?q=21100900151&tip=sid)]

---

## Summer Schools

> Summer schools are one of the best ways to get intensive, structured exposure to current CV research. Most run annually and accept applications from MSc students, PhD students, postdocs, and industry researchers. 

> Status: ✅ active (running regularly) · 🗄️ concluded (no longer running)

* [ICVSS](https://iplab.dmi.unict.it/icvss/): International Computer Vision Summer School [2007-Present], Sicily, Italy · competitive application · winner of the IEEE PAMI Mark Everingham Prize (2017) · ✅ active
* [BMVA CVSS](https://cvss.bmva.org/): British Computer Vision Summer School [2013-Present], UK · Organized by BMVA · ✅ active
* [VISUM](https://visum.inesctec.pt): Machine Intelligence and Visual Computing Summer School [2013-2020], Porto, Portugal · 🗄️ concluded

---

## Popular Articles

* Foundational Must-Reads
  > Ten papers every computer vision researcher should know. These defined the field's trajectory and are cited in virtually every modern CV paper.

  * [Backprop, 1986] Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams. "Learning representations by back-propagating errors." Nature 323 (1986): 533-536. [[paper](https://doi.org/10.1038/323533a0)]
  * [LeNet-5, 1998] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998). [[paper](https://doi.org/10.1109/5.726791)] — established CNNs as the standard for visual recognition
  * [SIFT, 2004] Lowe, David G. "Distinctive image features from scale-invariant keypoints." IJCV 60.2 (2004): 91-110. [[paper](https://doi.org/10.1023/B:VISI.0000029664.99615.94)] — the dominant feature descriptor for a decade
  * [BoVW, 2003/2004] Sivic, and Zisserman. "Video Google: A text retrieval approach to object matching in videos." Proceedings ninth IEEE international conference on computer vision. IEEE, 2003. Csurka, Gabriella, et al. "Visual categorization with bags of keypoints." Workshop on statistical learning in computer vision, ECCV. Vol. 1. No. 1-22. 2004. [[paper](https://doi.org/10.1109/ICCV.2003.1238663)] — introduced the bag-of-visual-words framework using visual vocabularies for image classification
  * [HOG, 2005] Dalal, Navneet, and Bill Triggs. "Histograms of oriented gradients for human detection." CVPR (2005). [[paper](https://doi.org/10.1109/CVPR.2005.177)] — foundation of pedestrian and object detection
  * [ImageNet, 2009] Deng, Jia, et al. "ImageNet: A large-scale hierarchical image database." CVPR (2009). [[paper](https://doi.org/10.1109/CVPR.2009.5206848)] — the benchmark that enabled the deep learning era
  * [AlexNet, 2012] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet classification with deep convolutional neural networks." NeurIPS (2012). [[paper](https://doi.org/10.1145/3065386)] — the paper that started the deep learning era in CV
  * [GAN, 2014] Goodfellow, Ian, et al. "Generative adversarial nets." NeurIPS (2014). [[paper](https://doi.org/10.1145/3422622)] — introduced the GAN framework that underpins generative CV
  * [U-Net, 2015] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-Net: Convolutional networks for biomedical image segmentation." MICCAI (2015). [[paper](https://doi.org/10.1007/978-3-319-24574-4_28)] — the default architecture for segmentation tasks
  * [ResNet, 2016] He, Kaiming, et al. "Deep residual learning for image recognition." CVPR (2016). [[paper](https://doi.org/10.1109/IEEESTD.2001.92771)] — residual connections solved the vanishing gradient problem; still the most-used backbone
  * [Attention, 2017] Vaswani, Ashish, et al. "Attention is all you need." NeurIPS (2017). [[paper](https://dl.acm.org/doi/10.5555/3295222.3295349)] — the transformer architecture that ViT and every modern foundation model is built on
  * [ViT, 2020] Dosovitskiy, Alexey, et al. "An image is worth 16x16 words." ICLR (2021). [[paper](https://doi.org/10.48550/arXiv.2010.11929)] — brought transformers to vision and reshaped every sub-field
* Object Classification
  * [LeNet-5, 1998] LeCun, Yann, et al. "Gradient-based learning applied to document recognition." Proceedings of the IEEE 86.11 (1998): 2278-2324.
  * [AlexNet, 2012] Krizhevsky, Alex, Ilya Sutskever, and Geoffrey E. Hinton. "Imagenet classification with deep convolutional neural networks." Advances in neural information processing systems 25 (2012).
  * [ZFNet, 2014] Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part I 13. Springer International Publishing, 2014.
  * [VGG, 2014] Simonyan, Karen and Andrew Zisserman. “Very Deep Convolutional Networks for Large-Scale Image Recognition.” CoRR abs/1409.1556 (2014): n. Pag.
  * [GoogLeNet, 2015] Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
  * [ResNet, 2016] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
  * [InceptionV3, 2016] Szegedy, Christian, et al. "Rethinking the inception architecture for computer vision." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
  * [Xception, 2017] Chollet, François. "Xception: Deep learning with depthwise separable convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
  * [EfficientNet, 2019] Tan, Mingxing, and Quoc Le. "Efficientnet: Rethinking model scaling for convolutional neural networks." International conference on machine learning. PMLR, 2019.
  * [ViT, 2020] Dosovitskiy, Alexey, et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale." International Conference on Learning Representations. 2020.
  * [ConvNeXt, 2022] Liu, Zhuang et al. “A ConvNet for the 2020s.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 11966-11976.
* Object Classification - Lightweight
  * [SqueezeNet, 2016] Iandola, Forrest N., et al. "SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and< 0.5 MB model size." arXiv preprint arXiv:1602.07360 (2016).
  * [MobileNetV2, 2018] Sandler, Mark, et al. "Mobilenetv2: Inverted residuals and linear bottlenecks." Proceedings of the IEEE conference on computer vision and pattern recognition. 2018.  
  * [ShuffleNetV2, 2018] Ma, Ningning, et al. "Shufflenet v2: Practical guidelines for efficient cnn architecture design." Proceedings of the European conference on computer vision (ECCV). 2018.
  * [MobileNetV3, 2019] Howard, Andrew, et al. "Searching for mobilenetv3." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
  * [GhostNetV1, 2020] Han, Kai, et al. "Ghostnet: More features from cheap operations." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
  * [MobileViT, 2021] Mehta, Sachin, and Mohammad Rastegari. "Mobilevit: light-weight, general-purpose, and mobile-friendly vision transformer." arXiv preprint arXiv:2110.02178 (2021).
  * [GhostNetV2, 2022] Tang, Yehui, et al. "GhostNetv2: enhance cheap operation with long-range attention." Advances in Neural Information Processing Systems 35 (2022): 9969-9982.
  * [ConvNeXt-Tiny, 2022] Liu, Zhuang et al. “A ConvNet for the 2020s.” 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) (2022): 11966-11976.
  * [MaxViT-Tiny, 2022] Tu, Zhengzhong, et al. "Maxvit: Multi-axis vision transformer." European conference on computer vision. Cham: Springer Nature Switzerland, 2022.
  * [MobileFormer, 2022] Chen, Yinpeng, et al. "Mobile-former: Bridging mobilenet and transformer." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
  * [ConvNeXtV2-Tiny, 2023] Woo, Sanghyun, et al. "Convnext v2: Co-designing and scaling convnets with masked autoencoders." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
  * [Mobileone, 2023] Vasu, Pavan Kumar Anasosalu, et al. "Mobileone: An improved one millisecond mobile backbone." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023.
  * [TinyViM, 2025] Ma, Xiaowen, Zhenliang Ni, and Xinghao Chen. "Tinyvim: Frequency decoupling for tiny hybrid vision mamba." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2025.
  * [SeaFormer++, 2025] Wan, Qiang, et al. "SeaFormer++: Squeeze-enhanced axial transformer for mobile visual recognition." International Journal of Computer Vision 133.6 (2025): 3645-3666.
* Object Detection
  * [Faster R-CNN, 2015] Ren, Shaoqing, et al. "Faster r-cnn: Towards real-time object detection with region proposal networks." Advances in neural information processing systems 28 (2015).
  * [SSD, 2016] Liu, Wei, et al. "Ssd: Single shot multibox detector." Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11–14, 2016, Proceedings, Part I 14. Springer International Publishing, 2016.
  * [RetinaNet, 2017] Lin, Tsung-Yi, et al. "Focal loss for dense object detection." Proceedings of the IEEE international conference on computer vision. 2017.
  * [YOLOV3, 2018] Redmon, Joseph, and Ali Farhadi. "Yolov3: An incremental improvement." arXiv preprint arXiv:1804.02767 (2018).
  * [YOLOX, 2021] Ge, Zheng, et al. "Yolox: Exceeding yolo series in 2021." arXiv preprint arXiv:2107.08430 (2021).
  * [YOLOR, 2021] Wang, Chien-Yao, I-Hau Yeh, and Hong-Yuan Mark Liao. "You only learn one representation: Unified network for multiple tasks." arXiv preprint arXiv:2105.04206 (2021).
  * [YOLOV7, 2023] Wang, Chien-Yao, Alexey Bochkovskiy, and Hong-Yuan Mark Liao. "YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
* Object Segmentation - Semantic / Instance / Panoptic
  * Classical: Graph Cut / Normalized Cut, Fuzzy Clustering, Mean-shift / Quick-shift, SLIC, Active Contours (Snakes), Region Growing, K-means Clustering, Watershed, Level Set Methods, Markov Random Fields (MRF), Edge (1st / 2nd derivatives) + filling.
  * [U-Net, 2015] Ronneberger, Olaf, Philipp Fischer, and Thomas Brox. "U-net: Convolutional networks for biomedical image segmentation." Medical Image Computing and Computer-Assisted Intervention–MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18. Springer International Publishing, 2015.
  * [DeepLabV3, 2017] Chen, Liang-Chieh, et al. "Rethinking atrous convolution for semantic image segmentation." arXiv preprint arXiv:1706.05587 (2017).
  * [PSPNet, 2017] Zhao, Hengshuang, et al. "Pyramid scene parsing network." Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.
  * [Mask R-CNN, 2017] He, Kaiming, et al. "Mask r-cnn." Proceedings of the IEEE international conference on computer vision. 2017.
  * [U-Net++, 2018] Zhou, Zongwei et al. “UNet++: A Nested U-Net Architecture for Medical Image Segmentation.” Deep Learning in Medical Image Analysis and Multimodal Learning for Clinical Decision Support : 4th International Workshop, DLMIA 2018, and 8th International Workshop, ML-CDS 2018, held in conjunction with MICCAI 2018, Granada, Spain, S... 11045 (2018): 3-11.
  * [DeepLabV3+, 2018] Chen, Liang-Chieh et al. “Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation.” European Conference on Computer Vision (2018).
  * [MaskFormer, 2021] Cheng, Bowen, Alex Schwing, and Alexander Kirillov. "Per-pixel classification is not all you need for semantic segmentation." Advances in Neural Information Processing Systems 34 (2021): 17864-17875.
  * [SegFormer, 2021] E. Xie, W. Wang, Z. Yu, A. Anandkumar, J. M. Alvarez, and P. Luo, “Segformer: Simple and efficient design for semantic segmentation with transformers,” Advances in neural information processing systems, vol. 34, pp. 12 077–12 090, 2021.
  * [SAM, 2023] A. Kirillov, E. Mintun, N. Ravi, et al., “Segment anything,” in Proceedings of the IEEE/CVF International Conference on Computer Vision, 2023, pp. 4015–4026.
  * [SEEM, 2023] Zou, Xueyan, et al. "Segment everything everywhere all at once." Advances in neural information processing systems 36 (2023): 19769-19782.
* Feature Matching
  * {Local Features} [Superpoint, 2018] DeTone, Daniel, Tomasz Malisiewicz, and Andrew Rabinovich. "Superpoint: Self-supervised interest point detection and description." Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2018.
  * {Local Features} [D2-Net, 2019] Dusmanu, Mihai, et al. "D2-net: A trainable cnn for joint detection and description of local features." arXiv preprint arXiv:1905.03561 (2019).
  * [R2D2, 2019] Revaud, Jerome, et al. "R2D2: repeatable and reliable detector and descriptor." arXiv preprint arXiv:1906.06195 (2019).
  * {Detector-Based Matcher} [SuperGlue, 2020] Sarlin, Paul-Edouard, et al. "Superglue: Learning feature matching with graph neural networks." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2020.
  * {Detector-Free Matcher} [DRC-Net, 2020] Li, Xinghui, et al. "Dual-resolution correspondence networks." Advances in Neural Information Processing Systems 33 (2020): 17346-17357.
  * {Local Features} [DISK, 2020] Tyszkiewicz, Michał, Pascal Fua, and Eduard Trulls. "DISK: Learning local features with policy gradient." Advances in Neural Information Processing Systems 33 (2020): 14254-14265.
  * {Detector-Free Matcher} [LoFTR, 2021] Sun, Jiaming, et al. "LoFTR: Detector-free local feature matching with transformers." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.
  * {Detector-Free Matcher} [MatchFormer, 2022] Wang, Qing, et al. "Matchformer: Interleaving attention in transformers for feature matching." Proceedings of the Asian Conference on Computer Vision. 2022.
  * {Detector-Based Matcher} [LightGlue, 2023] Lindenberger, Philipp, Paul-Edouard Sarlin, and Marc Pollefeys. "LightGlue: Local Feature Matching at Light Speed." arXiv preprint arXiv:2306.13643 (2023).
  * {Detector-Based Matcher} [GlueStick, 2023] Pautrat, Rémi, et al. "Gluestick: Robust image matching by sticking points and lines together." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
  * {Detector-Free Matcher} [OAMatcher, 2023] Dai, Kun, et al. "OAMatcher: An Overlapping Areas-based Network for Accurate Local Feature Matching." arXiv preprint arXiv:2302.05846 (2023).
  * Edstedt, Johan, et al. "RoMa: Revisiting Robust Losses for Dense Feature Matching." arXiv preprint arXiv:2305.15404 (2023).
  * Shen, Xuelun, et al. "GIM: Learning Generalizable Image Matcher From Internet Videos." The Twelfth International Conference on Learning Representations. 2023.
  * {Detector-Free Matcher} [DeepMatcher, 2024] Xie, Tao, et al. "Deepmatcher: a deep transformer-based network for robust and accurate local feature matching." Expert Systems with Applications 237 (2024): 121361.
  * {Detector-Free Matcher} [XFeat, 2024] Potje, Guilherme, et al. "XFeat: Accelerated Features for Lightweight Image Matching." IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR). 2024.
* Object Tracking
  * [SORT, 2017] Wojke, Nicolai, Alex Bewley, and Dietrich Paulus. "Simple online and realtime tracking with a deep association metric." 2017 IEEE international conference on image processing (ICIP). IEEE, 2017.
  * [Tracktor, 2019] Bergmann, Philipp, Tim Meinhardt, and Laura Leal-Taixe. "Tracking without bells and whistles." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2019.
  * [FairMOT, 2021] Zhang, Yifu, et al. "Fairmot: On the fairness of detection and re-identification in multiple object tracking." International Journal of Computer Vision 129 (2021): 3069-3087.
  * [STARK, 2021] Yan, Bin, et al. "Learning spatio-temporal transformer for visual tracking." Proceedings of the IEEE/CVF international conference on computer vision. 2021.
  * [MixFormer, 2022] Cui, Yutao, et al. "Mixformer: End-to-end tracking with iterative mixed attention." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
  * [ByteTrack, 2022] Zhang, Yifu, et al. "Bytetrack: Multi-object tracking by associating every detection box." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2022.
* Pose Estimation
  * Classical: Active Shape Models (ASM), Active Appearance Models (AAM), Pictorial Structures, Deformable Part Models (DPM).
  * [DeepPose, 2014] Toshev, Alexander, and Christian Szegedy. "DeepPose: Human pose estimation via deep neural networks." CVPR (2014).
  * [Stacked Hourglass, 2016] Newell, Alejandro, Kaiyu Yang, and Jia Deng. "Stacked hourglass networks for human pose estimation." ECCV (2016).
  * [OpenPose, 2019] Cao, Zhe, et al. "OpenPose: Realtime multi-person 2D pose estimation using part affinity fields." IEEE TPAMI (2019).
  * [HRNet, 2019] Wang, Jingdong, et al. "Deep high-resolution representation learning for visual recognition." IEEE TPAMI (2019).
  * [ViTPose, 2022] Xu, Yufei, et al. "ViTPose: Simple vision transformer baselines for human pose estimation." NeurIPS (2022).
  * [DWPose, 2023] Yang, Tianhao, et al. "Effective whole-body pose estimation with two-stages distillation." ICCV Workshop (2023).
  * [RTMPose, 2023] Jiang, Tao, et al. "RTMPose: Real-time multi-person pose estimation based on RTMDet." arXiv (2023).
  * [UniPose, 2024] Yang, Junjie, et al. "UniPose: Detecting any keypoints." CVPR (2024).
* Depth Estimation
  * Classical: stereo matching, structured light, time-of-flight (ToF), SfM (Structure from Motion).
  * [Make3D, 2009] Saxena, Ashutosh, Min Sun, and Andrew Y. Ng. "Make3D: Learning 3D scene structure from a single still image." IEEE TPAMI (2009).
  * [Eigen et al., 2014] Eigen, David, Christian Puhrsch, and Rob Fergus. "Depth map prediction from a single image using a multi-scale deep network." NeurIPS (2014).
  * [DenseDepth, 2018] Alhashim, Ibraheem, and Peter Wonka. "High quality monocular depth estimation via transfer learning." arXiv (2018).
  * [MiDaS, 2020] Ranftl, René, et al. "Towards robust monocular depth estimation: Mixing datasets for zero-shot cross-dataset transfer." IEEE TPAMI (2020).
  * [AdaBins, 2021] Bhat, Shariq Farooq, et al. "AdaBins: Depth estimation using adaptive bins." CVPR (2021).
  * [DPT, 2021] Ranftl, René, et al. "Vision transformers for dense prediction." ICCV (2021).
  * [ZoeDepth, 2023] Bhat, Shariq Farooq, et al. "ZoeDepth: Zero-shot transfer by combining relative and metric depth." arXiv (2023).
  * [Depth Anything, 2024] Yang, Lihe, et al. "Depth anything: Unleashing the power of large-scale unlabeled data." CVPR (2024).
  * [Depth Anything V2, 2024] Yang, Lihe, et al. "Depth Anything V2." NeurIPS (2024).
  * [Marigold, 2024] Ke, Bingxin, et al. "Repurposing diffusion-based image generators for monocular depth estimation." CVPR (2024).
* Media Generation
  * [DCGAN, 2015] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
  * [BigGAN, 2018] Brock, Andrew, Jeff Donahue, and Karen Simonyan. "Large scale GAN training for high fidelity natural image synthesis." arXiv preprint arXiv:1809.11096 (2018).
  * [StyleGANv3, 2021] Karras, Tero, et al. "Alias-free generative adversarial networks." Advances in Neural Information Processing Systems 34 (2021): 852-863.
  * [DALL-E, 2021] Ramesh, Aditya, et al. "Zero-shot text-to-image generation." International conference on machine learning. Pmlr, 2021.
  * [LAFITE, 2021] Zhou, Y., et al. "Lafite: Towards language-free training for text-to-image generation. arxiv 2021." arXiv preprint arXiv:2111.13792 2 (2021).
  * [CLIP, 2021] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." International conference on machine learning. PMLR, 2021.
  * [Imagen, 2022] Saharia, Chitwan, et al. "Photorealistic text-to-image diffusion models with deep language understanding." Advances in neural information processing systems 35 (2022): 36479-36494.
  * [GLIDE, 2022] Nichol, Alexander Quinn, et al. "GLIDE: Towards Photorealistic Image Generation and Editing with Text-Guided Diffusion Models." International Conference on Machine Learning. PMLR, 2022.
  * [unCLIP, 2022] Ramesh, Aditya, et al. "Hierarchical Text-Conditional Image Generation with CLIP Latents." arXiv preprint arXiv:2204.06125 (2022).
  * [LDM / Stable Diffusion (SD), 2022] Rombach, Robin, et al. "High-resolution image synthesis with latent diffusion models." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
  * [DALL-E 2, 2022] Ramesh, Aditya, et al. "Hierarchical text-conditional image generation with clip latents." arXiv preprint arXiv:2204.06125 1.2 (2022).
  * [DALL-E 3, 2023] Betker, James, et al. "Improving image generation with better captions." Computer Science. <https://cdn.openai.com/papers/dall-e-3.pdf> 2.3 (2023): 8.
  * [SDXL, 2023] Podell, Dustin, et al. "SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis." The Twelfth International Conference on Learning Representations. 2023.
* Vision-Language Models (VLMs)
  * [CLIP, 2021] Radford, Alec, et al. "Learning transferable visual models from natural language supervision." ICML (2021).
  * [ALIGN, 2021] Jia, Chao, et al. "Scaling up visual and vision-language representation learning with noisy text supervision." ICML (2021).
  * [BLIP, 2022] Li, Junnan, et al. "BLIP: Bootstrapping language-image pre-training for unified vision-language understanding and generation." ICML (2022).
  * [Flamingo, 2022] Alayrac, Jean-Baptiste, et al. "Flamingo: a visual language model for few-shot learning." NeurIPS (2022).
  * [BLIP-2, 2023] Li, Junnan, et al. "BLIP-2: Bootstrapping language-image pre-training with frozen image encoders and large language models." ICML (2023).
  * [LLaVA, 2023] Liu, Haotian, et al. "Visual instruction tuning." NeurIPS (2023).
  * [InstructBLIP, 2023] Dai, Wenliang, et al. "InstructBLIP: Towards general-purpose vision-language models with instruction tuning." NeurIPS (2023).
  * [GPT-4V, 2023] OpenAI. "GPT-4 technical report." arXiv (2023).
  * [LLaVA-1.5, 2023] Liu, Haotian, et al. "Improved baselines with visual instruction tuning." CVPR (2024).
  * [Qwen-VL, 2023] Bai, Jinze, et al. "Qwen-VL: A versatile vision-language model for understanding, localization, text reading, and beyond." arXiv (2023).
* Image Retrieval
  * [LSMH, 2016] Lu, Xiaoqiang, Xiangtao Zheng, and Xuelong Li. "Latent semantic minimal hashing for image retrieval." IEEE Transactions on Image Processing 26.1 (2016): 355-368.
  * [R–GeM, 2018] Radenović, Filip, Giorgos Tolias, and Ondřej Chum. "Fine-tuning CNN image retrieval with no human annotation." IEEE transactions on pattern analysis and machine intelligence 41.7 (2018): 1655-1668.
  * [HOW, 2020] Tolias, Giorgos, Tomas Jenicek, and Ondřej Chum. "Learning and aggregating deep local descriptors for instance-level recognition." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part I 16. Springer International Publishing, 2020.
  * [DELG, 2020] Cao, Bingyi, Andre Araujo, and Jack Sim. "Unifying deep local and global features for image search." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XX 16. Springer International Publishing, 2020.
  * [SOLAR, 2020] Ng, Tony, et al. "SOLAR: second-order loss and attention for image retrieval." Computer Vision–ECCV 2020: 16th European Conference, Glasgow, UK, August 23–28, 2020, Proceedings, Part XXV 16. Springer International Publishing, 2020.
  * [FIRe, 2021] Weinzaepfel, Philippe, et al. "Learning Super-Features for Image Retrieval." International Conference on Learning Representations. 2021.
  * [DOLG, 2021] Yang, Min, et al. "Dolg: Single-stage image retrieval with deep orthogonal fusion of local and global features." Proceedings of the IEEE/CVF International conference on Computer Vision. 2021.
  * [Token, 2022] Wu, Hui, et al. "Learning token-based representation for image retrieval." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 36. No. 3. 2022.
  * [CVNet, 2022] Lee, Seongwon, et al. "Correlation verification for image retrieval." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.
  * [GLAM, 2022] Song, Chull Hwan, Hye Joo Han, and Yannis Avrithis. "All the attention you need: Global-local, spatial-channel attention for image retrieval." Proceedings of the IEEE/CVF winter conference on applications of computer vision. 2022.
  * [SuperGlobal, 2023] Shao, Shihao, et al. "Global features are all you need for image retrieval and reranking." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
  * [CFCD, 2023] Zhu, Yunquan, et al. "Coarse-to-fine: Learning compact discriminative representation for single-stage image retrieval." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.
  * [SENet, 2023] Lee, Seongwon, et al. "Revisiting self-similarity: Structural embedding for image retrieval." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.
  * [CiDeR, 2024] Song, Chull Hwan, et al. "On train-test class overlap and detection for image retrieval." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.
* WIP:
  * Image Super-Resolution / Image Restoration
  * Saliency Detection
  * Vanishing Point Detection
  * Image Colorization
  * Image Captioning
  * Video Summarization and Captioning
  * Explainable AI (XAI)
  * Text Recognition
  * Data Compression
  * Affective Computing
  * Virtual reality (VR)
  * Augmented reality (AR)
  * Visual Question Answering (VQA)
  * DeepFake Detection
  * 3D Reconstruction
  * Biometric Analysis
  * Meta Learning
  * Semi-Supervised Learning - Zero/One/Few shot

---

## Reference Books

| Book | Links |
| --------------- | --------------- |
| Antonio Torralba, Phillip Isola, William T. Freeman. “Foundations of Computer Vision” MIT Press, (2024). | [goodreads](https://www.goodreads.com/book/show/157976035-foundations-of-computer-vision?from_search=true&from_srp=true&qid=y0fzNP4eVX&rank=2) |
| Nixon, Mark, and Alberto Aguado. “Feature extraction and image processing for computer vision” Academic press, (2019). | [goodreads](https://www.goodreads.com/book/show/14788673-feature-extraction-and-image-processing-for-computer-vision) |
| González, Rafael Corsino and Richard E. Woods. “Digital image processing, 4th Edition” (2018). | [goodreads](https://www.goodreads.com/book/show/42937189-digital-image-processing) |
| E.R. Davies. “Computer Vision: Principles, Algorithms, Applications, Learning” Academic press, (2017). | [goodreads](https://www.goodreads.com/book/show/36987287-computer-vision) |
| Prince, Simon. “Computer Vision: Models, Learning, and Inference” (2012). | [goodreads](https://www.goodreads.com/book/show/15792261-computer-vision) |
| Forsyth, David Alexander and Jean Ponce. “Computer Vision - A Modern Approach, Second Edition” (2011). |[goodreads](https://www.goodreads.com/book/show/14857613-computer-vision) |
| Szeliski, Richard. “Computer Vision - Algorithms and Applications” Texts in Computer Science (2010). | [goodreads](https://www.goodreads.com/book/show/9494221-computer-vision) |
| Bishop, Charles M.. “Pattern recognition and machine learning, 5th Edition” Information science and statistics (2007). | [goodreads](https://www.goodreads.com/book/show/37572203-pattern-recognition-and-machine-learning) |
| Harltey, Andrew and Andrew Zisserman. “Multiple view geometry in computer vision (2. ed.)” (2003). | [goodreads](https://www.goodreads.com/book/show/89897.Multiple_View_Geometry_in_Computer_Vision) |
| Stockman, George C. and Linda G. Shapiro. “Computer Vision” (2001). | [goodreads](https://www.goodreads.com/book/show/19371156-computer-vision) |

---

## Courses

| Course | Year | Instructor | Source |
| --------------- | --------------- | --------------- | --------------- |
| [Introduction to Computer Vision](https://browncsci1430.github.io) | 2026 | James Tompkin | Brown |
| [Deep Learning for Computer Vision](http://cs231n.stanford.edu) | 2025 | Fei-Fei Li | Stanford |
| [Advances in Computer Vision](http://6.8300.csail.mit.edu/sp23/) | 2023 | William T. Freeman | MIT |
| [OpenCV for Python Developers](https://www.linkedin.com/learning/opencv-for-python-developers) | 2023 | Patrick Crawford | LinkedIn Learning |
| [Computer Vision](https://www.youtube.com/playlist?list=PL05umP7R6ij35L2MHGzis8AEHz7mg381_)| 2021 | Andreas Geiger | University of Tübingen |
| [Computer Vision](https://www.youtube.com/playlist?list=PLd3hlSJsX_IkXSinyREhlMjFvpNfpazfN) | 2021 | Yogesh S Rawat / Mubarak Shah | University of Central Florida |
| [Advanced Computer Vision](https://www.youtube.com/playlist?list=PLd3hlSJsX_Ilwca04yxhrjcdzx7BS2vDh) | 2021| Mubarak Shah | University of Central Florida |
| [Deep Learning for Computer Vision](https://www.youtube.com/playlist?list=PL5-TkQAfAZFbzxjBHtzdVCWE0Zbhomg7r) | 2020 | Justin Johnson | University of Michigan |
| [Advanced Deep Learning for Computer Vision](https://www.youtube.com/playlist?list=PLog3nOPCjKBnjhuHMIXu4ISE4Z4f2jm39)| 2020 | Laura Leal-Taixé / Matthias Niessner  | Technical University of Munich |
| [Introduction to Digital Image Processing](https://www.youtube.com/playlist?list=PL2mBI0yFsKk-p73KQ4iPdsi10hQC4Zd-0)| 2020 | Ahmadreza Baghaie | New York Institute of Technology|
| [Quantitative Imaging](https://www.youtube.com/playlist?list=PLTWuXgjdOrnmXVVQG5DRkVeOIGOcTmCIw) | 2019 | Kevin Mader | ETH Zurich |
| [Convolutional Neural Networks for Visual Recognition](https://www.youtube.com/playlist?list=PLf7L7Kg8_FNxHATtLwDceyh72QQL9pvpQ) | 2017 | Fei-Fei Li | Stanford University  |
| [Introduction to Digital Image Processing](https://www.youtube.com/playlist?list=PLuh62Q4Sv7BUf60vkjePfcOQc8sHxmnDX) | 2015|Rich Radke | Rensselaer Polytechnic Institute|
| [Machine Learning for Robotics and Computer Vision](https://www.youtube.com/playlist?list=PLTBdjV_4f-EIiongKlS9OKrBEp8QR47Wl) | 2014| Rudolph Triebel |  Technical University of Munich |
| [Multiple View Geometry](https://www.youtube.com/playlist?list=PLTBdjV_4f-EJn6udZ34tht9EVIW7lbeo4) | 2013 | Daniel Cremers | Technical University of Munich |
| [Variational Methods for Computer Vision](https://www.youtube.com/playlist?list=PLTBdjV_4f-EJ7A2iIH5L5ztqqrWYjP2RI) | 2013 | Daniel Cremers | Technical University of Munich |
| [Computer Vision](https://www.youtube.com/playlist?list=PLd3hlSJsX_ImKP68wfKZJVIPTd8Ie5u-9) | 2012| Mubarak Shah | University of Central Florida |
| [Image and video processing](https://www.youtube.com/playlist?list=PLZ9qNFMHZ-A79y1StvUUqgyL-O0fZh2rs) | - | Guillermo Sapiro | Duke University|
| [Introduction to Computer Vision](https://www.udacity.com/course/introduction-tocomputer-vision--ud810) | - | Aaron Bobick / Irfan Essa | Udacity |

---

## Repos

> Tags: Object Classification `[ObjCls]`, Object Detection `[ObjDet]`, Object Segmentation `[ObjSeg]`, General Library `[GenLib]`, Text Reading / Object Character Recognition `[OCR]`, Action Recognition `[ActRec]`, Object Tracking `[ObjTrk]`, Data Augmentation `[DatAug]`, Simultaneous Localization and Mapping `[SLAM]`, Outlier/Anomaly/Novelty Detection `[NvlDet]`, Content-based Image Retrieval `[CBIR]`, Image Enhancement `[ImgEnh]`, Aesthetic Assessment `[AesAss]`, Explainable Artificial Intelligence `[XAI]`, Text-to-Image Generation `[TexImg]`, Pose Estimation `[PosEst]`, Video Matting `[VidMat]`, Eye Tracking `[EyeTrk]`

| Repo | Tags | Description |
| --------------- | --------------- | --------------- |
| [computervision-recipes](https://github.com/microsoft/computervision-recipes) | `[GenLib]` | Microsoft, Best Practices, code samples, and documentation for Computer Vision |
| [FastAI](https://github.com/fastai/fastai) | `[GenLib]` | FastAI, Library over PyTorch used for learning and practicing machine learning and deep learning |
| [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning) | `[GenLib]` | PyTorchLightning, Lightweight PyTorch wrapper for high-performance AI research |
| [ignite](https://github.com/pytorch/ignite) | `[GenLib]` | PyTorch, High-level library to help with training and evaluating neural networks in PyTorch flexibly and transparently |
| [pytorch_geometric](https://github.com/pyg-team/pytorch_geometric) | `[GenLib]` | Graph Neural Network Library for PyTorch |
| [kornia](https://github.com/kornia/kornia) | `[GenLib]` | Open Source Differentiable Computer Vision Library |
| [ncnn](https://github.com/Tencent/ncnn) | `[GenLib]` | Tencent, High-performance neural network inference framework optimized for the mobile platform |
| [ITK](https://github.com/InsightSoftwareConsortium/ITK) | `[GenLib]` | open-source, cross-platform toolkit for N-dimensional scientific image processing, segmentation, and registration |
| [VTK](https://github.com/Kitware/VTK) | `[GenLib]` | open-source software system for image processing, 3D graphics, volume rendering and visualization |
| [MONAI](https://github.com/Project-MONAI/MONAI) | `[GenLib]` | PyTorch-based, open-source framework for deep learning in healthcare imaging |
| [MediaPipe](https://github.com/google/mediapipe) | `[ObjDet]` `[ObjSeg]` `[ObjTrk]` `[GenLib]` | Google, iOS - Andriod - C++ - Python - Coral, Face Detection - Face Mesh - Iris - Hands - Pose - Holistic - Hair Segmentation - Object Detection - Box Tracking - Instant Motion Tracking - Objectron - KNIFT (Similar to SIFT) |
| [PyTorch image models](https://github.com/rwightman/pytorch-image-models) | `[ObjCls]` | rwightman, PyTorch image classification models, scripts, pretrained weights |
| [mmclassification](https://github.com/open-mmlab/mmclassification) | `[ObjCls]` | OpenMMLab, Image Classification Toolbox and Benchmark |
| [vit-pytorch](https://github.com/lucidrains/vit-pytorch) | `[ObjCls]` | SOTA for vision transformers |
| [face_classification](https://github.com/oarriaga/face_classification) | `[ObjCls]` `[ObjDet]`| Real-time face detection and emotion/gender classification |
| [mmdetection](https://github.com/open-mmlab/mmdetection) | `[ObjDet]` | OpenMMLab, Image Detection Toolbox and Benchmark |
| [detectron2](https://github.com/facebookresearch/detectron2) | `[ObjDet]` `[ObjSeg]` | Facebook, FAIR's next-generation platform for object detection, segmentation and other visual recognition tasks |
| [detr](https://github.com/facebookresearch/detr) | `[ObjDet]` | Facebook, End-to-End Object Detection with Transformers |
| [libfacedetection](https://github.com/ShiqiYu/libfacedetection) | `[ObjDet]` | An open source library for face detection in images, speed: ~1000FPS |
| [FaceDetection-DSFD](https://github.com/Tencent/FaceDetection-DSFD) | `[ObjDet]` | Tencent, SOTA face detector |
| [object-Detection-Metrics](https://github.com/rafaelpadilla/Object-Detection-Metrics) | `[ObjDet]` | Most popular metrics used to evaluate object detection algorithms |
| [SAHI](https://github.com/obss/sahi) | `[ObjDet]` `[ObjSeg]` | A lightweight vision library for performing large scale object detection/ instance segmentation |
| [yolov5](https://github.com/ultralytics/yolov5) | `[ObjDet]` | ultralytics |
| [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet) [pjreddie/darknet](https://github.com/pjreddie/darknet) | `[ObjDet]` | YOLOv4 / Scaled-YOLOv4 / YOLOv3 / YOLOv2 |
| [U-2-Net](https://github.com/xuebinqin/U-2-Net) | `[ObjDet]` | ultralytics U^2-Net: Going Deeper with Nested U-Structure for Salient Object Detection |
| [segmentation_models.pytorch](https://github.com/qubvel/segmentation_models.pytorch) | `[ObjSeg]` | qubvel, PyTorch segmentation models with pretrained backbones |
| [mmsegmentation](https://github.com/open-mmlab/mmsegmentation) | `[ObjSeg]` | OpenMMLab, Semantic Segmentation Toolbox and Benchmark |
| [mmocr](https://github.com/open-mmlab/mmocr) | `[OCR]` | OpenMMLab, Text Detection, Recognition and Understanding Toolbox |
| [pytesseract](https://github.com/madmaze/pytesseract) | `[OCR]` | A Python wrapper for Google Tesseract |
| [EasyOCR](https://github.com/JaidedAI/EasyOCR) | `[OCR]` | Ready-to-use OCR with 80+ supported languages and all popular writing scripts including Latin, Chinese, Arabic, Devanagari, Cyrillic and etc |
| [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) | `[OCR]` | Practical ultra lightweight OCR system, support 80+ languages recognition, provide data annotation and synthesis tools, support training and deployment among server, mobile, embedded and IoT devices|
| [PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg) | `[ObjSeg]` | Easy-to-use image segmentation library with awesome pre-trained model zoo, supporting wide-range of practical tasks in Semantic Segmentation, Interactive Segmentation, Panoptic Segmentation, Image Matting, 3D Segmentation, etc|
| [mmtracking](https://github.com/open-mmlab/mmtracking) | `[ObjTrk]` | OpenMMLab, Video Perception Toolbox for object detection and tracking |
| [mmaction](https://github.com/open-mmlab/mmaction) | `[ActRec]` | OpenMMLab, An open-source toolbox for action understanding based on PyTorch |
| [albumentations](https://github.com/albumentations-team/albumentations) | `[DatAug]` | Fast image augmentation library and an easy-to-use wrapper around other libraries |
| [ORB_SLAM2](https://github.com/raulmur/ORB_SLAM2) | `[SLAM]` | Real-Time SLAM for Monocular, Stereo and RGB-D Cameras, with Loop Detection and Relocalization Capabilities |
| [pyod](https://github.com/yzhao062/pyod) | `[NvlDet]` | Python Toolbox for Scalable Outlier Detection (Anomaly Detection) |
| [imagededup](https://github.com/idealo/imagededup) | `[CBIR]` | Image retrieval, CBIR, Find duplicate images made easy! |
| [image-match](https://github.com/ProvenanceLabs/image-match) | `[CBIR]` | Image retrieval, CBIR, Quickly search over billions of images |
| [Bringing-Old-Photos-Back-to-Life](https://github.com/microsoft/Bringing-Old-Photos-Back-to-Life) | `[ImgEnh]` | Microsoft, Bringing Old Photo Back to Life (CVPR 2020 oral) |
| [image-quality-assessment](https://github.com/idealo/image-quality-assessment) | `[AesAss]` | Idealo, Image Aesthetic, NIMA model to predict the aesthetic and technical quality of images |
| [aesthetics](https://github.com/ylogx/aesthetics) | `[AesAss]` | Image Aesthetics Toolkit using Fisher Vectors |
| [pytorch-cnn-visualizations](https://github.com/utkuozbulak/pytorch-cnn-visualizations) | `[XAI]` | Pytorch implementation of convolutional neural network visualization techniques |
| [DALLE2-pytorch](https://github.com/lucidrains/DALLE2-pytorch) | `[TexImg]` | Implementation of DALL-E 2, OpenAI's updated text-to-image synthesis neural network, in Pytorch |
| [imagen-pytorch](https://github.com/lucidrains/imagen-pytorch) | `[TexImg]` | Implementation of Imagen, Google's Text-to-Image Neural Network, in Pytorch |
| [openpose](https://github.com/CMU-Perceptual-Computing-Lab/openpose)| `[PosEst]` | OpenPose: Real-time multi-person keypoint detection library for body, face, hands, and foot estimation |
| [RobustVideoMatting](https://github.com/PeterL1n/RobustVideoMatting) | `[VidMat]` | Robust Video Matting in PyTorch, TensorFlow, TensorFlow.js, ONNX, CoreML! |
| [fastudp](https://github.com/visual-layer/fastdup) | `[NvlDet]` `[CBIR]` | An unsupervised and free tool for image and video dataset analysis |
| [Random-Erasing](https://github.com/zhunzhong07/Random-Erasing) | `[DatAug]` | Random Erasing Data Augmentation in PyTorch |
| [CutMix-PyTorch](https://github.com/clovaai/CutMix-PyTorch) | `[DatAug]` | Official Pytorch implementation of CutMix regularizer |
| [keras-cv](https://github.com/keras-team/keras-cv) | `[GenLib]` | Library of modular computer vision oriented Keras components |
| [PsychoPy](https://github.com/psychopy/psychopy) | `[EyeTrk]` | Library for running psychology and neuroscience experiments |
| [alibi-detect](https://github.com/SeldonIO/alibi-detect) | `[NvlDet]` | Algorithms for outlier, adversarial and drift detection |
| [Captum](https://github.com/pytorch/captum) | `[XAI]` | built by PyTorch team, Model interpretability and understanding for PyTorch |
| [Alibi](https://github.com/SeldonIO/alibi) | `[XAI]` | Algorithms for explaining machine learning models |
| [iNNvestigate](https://github.com/albermax/innvestigate) | `[XAI]` | for TF, A toolbox to iNNvestigate neural networks' predictions |
| [keras-vis](https://github.com/raghakot/keras-vis) | `[XAI]` | for Keras, Neural network visualization toolkit |
| [Keract](https://github.com/philipperemy/keract) | `[XAI]` | for Keras, Layers Outputs and Gradients |
| [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) | `[XAI]` | for PyTorch, Advanced AI Explainability for computer vision |
| [SHAP](https://github.com/shap/shap) | `[XAI]` | A game theoretic approach to explain the output of any machine learning model |
| [TensorWatch](https://github.com/microsoft/tensorwatch) | `[XAI]` | built by Microsoft, Debugging, monitoring and visualization for Python Machine Learning and Data Science |
| [WeightWatcher](https://github.com/CalculatedContent/WeightWatcher) | `[XAI]` | an open-source, diagnostic tool for analyzing Deep Neural Networks (DNN), without needing access to training or even test data |

---

## Dataset Collections

* [PyTorch - CV Datasets](https://pytorch.org/vision/stable/datasets.html), Meta
* [Tensorflow - CV Datasets](https://www.tensorflow.org/datasets/catalog/overview#image), Google
* [CVonline: Image Databases](https://homepages.inf.ed.ac.uk/rbf/CVonline/Imagedbase.htm), Edinburgh University, Thanks to Robert Fisher!
* [Kaggle](https://www.kaggle.com/datasets?tags=13207-Computer+Vision)
* [PaperWithCode](https://paperswithcode.com/area/computer-vision), Meta
* [RoboFlow](https://public.roboflow.com)
* [VisualData](https://visualdata.io/discovery)
* [CUHK Computer Vision](http://www.ee.cuhk.edu.hk/~xgwang/datasets.html)
* [VGG - University of Oxford](https://www.robots.ox.ac.uk/~vgg/data/)

---

## Annotation Tools

* [labelme](https://github.com/wkentaro/labelme), Image Polygonal Annotation with Python (polygon, rectangle, circle, line, point and image-level flag annotation).
* [CVAT](https://github.com/cvat-ai/cvat), an interactive video and image annotation tool for computer vision.
* [VoTT](https://github.com/microsoft/VoTT), Microsoft, Visual Object Tagging Tool: An electron app for building end to end Object Detection Models from Images and Videos.
* [labelImg](https://github.com/tzutalin/labelImg), Graphical image annotation tool and label object bounding boxes in images.
* [VIA](https://www.robots.ox.ac.uk/~vgg/software/via/), VGG Oxford, HTML-based standalone manual annotation software for image, audio and video.
* [FiftyOne](https://github.com/voxel51/fiftyone), open-source tool for building high-quality datasets and computer vision models.
* [makesense.ai](https://github.com/SkalskiP/make-sense), a free-to-use online tool for labeling photos.

---

## YouTube Channels

> Tags: Popular individuals `[Individual]`, Conference and event `[Conferences]`, University research groups `[University]`, Interactive talks and podscasts `[Talks]`, Research articles' explanation `[Papers]`.

* [@AurelienGeron](https://www.youtube.com/@AurelienGeron) `[Individual]`, Aurélien Géron: former lead of YouTube's video classification team, and author of the O'Reilly book Hands-On Machine Learning with Scikit-Learn and TensorFlow.
* [@howardjeremyp](https://www.youtube.com/@howardjeremyp) `[Individual]`, Jeremy Howard: former president and chief scientist of Kaggle, and co-founder of fast.ai.
* [@PieterAbbeel](https://www.youtube.com/@PieterAbbeel) `[Individual]`, Pieter Abbeel: professor of electrical engineering and computer sciences, University of California, Berkeley.
* [@pascalpoupart3507](https://www.youtube.com/@pascalpoupart3507) `[Individual]`, Pascal Poupart: professor in the David R. Cheriton School of Computer Science at the University of Waterloo.
* [@MatthiasNiessner](https://www.youtube.com/@MatthiasNiessner) `[Individual]`, Matthias Niessner: Professor at the Technical University of Munich and head of the Visual Computing Lab.
* [@MichaelBronsteinGDL](https://www.youtube.com/@MichaelBronsteinGDL) `[Individual]`, Michael Bronstein: DeepMind Professor of AI, University of Oxford / Head of Graph Learning Research, Twitter.
* [@DeepFindr](https://www.youtube.com/@DeepFindr) `[Individual]`, Videos about all kinds of Machine Learning / Data Science topics.
* [@deeplizard](https://www.youtube.com/@deeplizard) `[Individual]`, Videos about building collective intelligence.
* [@YannicKilcher](https://www.youtube.com/@YannicKilcher) `[Individual]`, Yannic Kilcher: videos about machine learning research papers, programming, and issues of the AI community, and the broader impact of AI in society.
* [@sentdex](https://www.youtube.com/@sentdex) `[Individual]`, sentdex: provides Python programming tutorials in machine learning, finance, data analysis, robotics, web development, game development and more.
* [@AAmini](https://www.youtube.com/@AAmini) `[Individual]`, Alexander Amini: Research Affilliate at MIT, videos about deep learning and data science.
* [@WhatsAI](https://www.youtube.com/@WhatsAI) `[Individual]`, Louis-François Bouchard: PhD in MILA, videos about AI.
* [mrdbourke](https://www.youtube.com/@mrdbourke) `[Individual]`, Daniel Bourke: ML engineer in healthcare, videos about AI.
* [marksaroufim](https://www.youtube.com/@marksaroufim) `[Individual]`, Mark Saroufim: PyTorch engineer at Meta (Facebook), videos about AI.
* [NicholasRenotte](https://www.youtube.com/@NicholasRenotte) `[Individual]`, Nicholas Renotte: videos about computer vision, natural language processign and reinforcement learning applications.
* [abhishekkrthakur](https://www.youtube.com/@abhishekkrthakur) `[Individual]`, Abhishek Thakur: world's first Quadruple Grand Master on Kaggle, videos about applied machine learning, deep learning, and data science.
* [@AladdinPersson](https://www.youtube.com/@AladdinPersson) `[Individual]`, Aladdin Persson: clear implementations of ML and CV papers from scratch in PyTorch and TensorFlow.
* [@CodeEmporium](https://www.youtube.com/@CodeEmporium) `[Individual]`, The Code Emporium: intuitive explanations of ML concepts and architectures.
* [@AICoffeeBreak](https://www.youtube.com/@AICoffeeBreak) `[Individual]`, AI Coffee Break with Letitia: short, accessible walkthroughs of recent AI and CV research.
* [@mildlyoverfitted](https://www.youtube.com/@mildlyoverfitted) `[Individual]`, Mildly Overfitted: hands-on CV and ML tutorials with clean code.
* [@SmithaKolan](https://www.youtube.com/@SmithaKolan) `[Individual]`, Smitha Kolan: computer vision tutorials focused on practical applications.
* [@KapilSachdeva](https://www.youtube.com/@KapilSachdeva) `[Individual]`, Kapil Sachdeva: in-depth explanations of ML research and engineering.
* [@alfcnz](https://www.youtube.com/@alfcnz) `[Individual]`, Alfredo Canziani: assistant professor at NYU, deep learning theory and practice.
* [@arp_ai](https://www.youtube.com/@arp_ai) `[Individual]`, Jay Alammar: applied ML and computer vision projects.
* [@bmvabritishmachinevisionas8529](https://www.youtube.com/@bmvabritishmachinevisionas8529) `[Conferences]`, BMVA: British Machine Vision Association.
* [@ComputerVisionFoundation](https://www.youtube.com/@ComputerVisionFoundation) `[Conferences]`, Computer Vision Foundation (CVF): co-sponsored conferences on computer vision (e.g. CVPR and ICCV).
* [@cvprtum](https://www.youtube.com/@cvprtum) `[University]`, Computer Vision Group at Technical University of Munich.
* [@UCFCRCV](https://www.youtube.com/@UCFCRCV) `[University]`, Center for Research in Computer Vision at University of Central Florida.
* [@dynamicvisionandlearninggr1022](https://www.youtube.com/@dynamicvisionandlearninggr1022) `[University]`, Dynamic Vision and Learning research group channel! Technical University of Munich.
* [@TubingenML](https://www.youtube.com/@TubingenML) `[University]`, Machine Learning groups at the University of Tübingen.
* [@computervisiontalks4659](https://www.youtube.com/@computervisiontalks4659) `[Talks]`, Computer Vision Talks.
* [@freecodecamp](https://www.youtube.com/@freecodecamp) `[Talks]`, Videos to learn how to code.
* [@LondonMachineLearningMeetup](https://www.youtube.com/@LondonMachineLearningMeetup) `[Talks]`, Largest machine learning community in Europe.
* [@LesHouches-iu6nv](https://www.youtube.com/@LesHouches-iu6nv) `[Talks]`, Summer school on Statistical Physics of Machine learning held in Les Houches, July 4 - 29, 2022.
* [@MachineLearningStreetTalk](https://www.youtube.com/@MachineLearningStreetTalk) `[Talks]`, top AI podcast on Spotify.
* [@WeightsBiases](https://www.youtube.com/@WeightsBiases) `[Talks]`, Weights and Biases team's conversations with industry experts, and researchers.
* [@PreserveKnowledge](https://www.youtube.com/@PreserveKnowledge/) `[Talks]`, Canada higher education media organization that focuses on advances in mathematics, computer science, and artificial intelligence.
* [@TwoMinutePapers](https://www.youtube.com/@TwoMinutePapers) `[Papers]`, Two Minute Papers: Explaining AI papers in few mins.
* [@TheAIEpiphany](https://www.youtube.com/@TheAIEpiphany) `[Papers]`, Aleksa Gordić: x-Google DeepMind, x-Microsoft engineer explaining AI papers.
* [@bycloudAI](https://www.youtube.com/@bycloudAI) `[Papers]`, bycloud: covers the latest AI tech/research papers for fun.

---

## Mailing Lists

* [Vision Science](http://visionscience.com/mailman/listinfo/visionlist_visionscience.com), announcements about industry/academic jobs in computer vision around the world (in English).
* [bull-i3](https://listes.irit.fr/sympa/info/bull-i3), posts about job opportunities in computer vision in France (in French).

---

## Curation Philosophy

Entries in this list are included because they are:

- **Genuinely educational** — they help you understand something, not just use it
- **Well-maintained** (or historically significant if archived)
- **Accessible** — free or widely available where possible

Entries marked `(last updated: YEAR)` in the libraries section are included for
historical or educational value despite no longer being actively developed.

This list is maintained by a computer vision researcher and university academic.
Suggestions and pull requests are welcome. Please check [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Thanks

* [Frida de Sigley](https://github.com/fdsig)
* [CORE Conference Ranking](http://portal.core.edu.au/conf-ranks/?search=4603&by=all&source=CORE2021&sort=arank&page=1)
* [Scimago Journal Ranking](https://www.scimagojr.com/journalrank.php)
* [benthecoder/yt-channels-DS-AI-ML-CS](https://github.com/benthecoder/yt-channels-DS-AI-ML-CS)
* [anomaly-detection-resources](https://github.com/yzhao062/anomaly-detection-resources), Anomaly detection related books, papers, videos, and toolboxes
* [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets) List of satellite image training datasets with annotations for computer vision and deep learning
* [awesome-Face_Recognition](https://github.com/ChanChiChoi/awesome-Face_Recognition), Computer vision papers about faces.
* [the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch), Curated list of tutorials, papers, projects, communities and more relating to PyTorch
