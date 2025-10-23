# Curated educational list for computer vision

[![Awesome](https://awesome.re/badge.svg)](https://github.com/mawady/awesome-cv)

---

## Contents

>
> * **[Python Libraries](#python-libraries)**
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
> * **[Misc](#misc)**
>

---

## Python Libraries

| Library | Description |
| --------------- | --------------- |
| [OpenCV](https://github.com/opencv/opencv) | Open Source Computer Vision Library|
| [Pillow](https://github.com/python-pillow/Pillow)| The friendly PIL fork (Python Imaging Library)|
| [scikit-image](https://github.com/scikit-image/scikit-image) | collection of algorithms for image processing|
| [SciPy](https://github.com/scipy/scipy)| open-source software for mathematics, science, and engineering|
| [mmcv](https://github.com/open-mmlab/mmcv)| OpenMMLab foundational library for computer vision research |
| [imutils](https://github.com/PyImageSearch/imutils) | A series of convenience functions to make basic image processing operations|
| [pgmagick](https://github.com/hhatto/pgmagick)| python based wrapper for GraphicsMagick/ImageMagick|
| [Mahotas](https://github.com/luispedro/mahotas) | library of fast computer vision algorithms (last updated: 2021)|
| [SimpleCV](https://github.com/sightmachine/SimpleCV#raspberry-pi) | The Open Source Framework for Machine Vision (last updated: 2015)|

---

## MATLAB Libraries

| Library | Description |
| --------------- | --------------- |
| [PMT](https://pdollar.github.io/toolbox/) | Piotr's Computer Vision Matlab Toolbox|
| [matlabfns](https://www.peterkovesi.com/matlabfns/)| MATLAB and Octave Functions for Computer Vision and Image Processing, P. Kovesi, University of Western Australia|
| [VLFeat](https://www.vlfeat.org/index.html) | open source library implements popular computer vision algorithms, A. Vedaldi and B. Fulkerson|
| [MLV](https://github.com/bwlabToronto/MLV_toolbox) | Mid-level Vision Toolbox (MLVToolbox), BWLab, University of Toronto|
| [ElencoCode](https://www.dropbox.com/s/bguw035yrqz0pwp/ElencoCode.docx?dl=0) | Loris Nanni's CV functions, University of Padova|

---

## Evaluation Metrics

* Performance - Classficiation
  * Confusion Matrix: TP, FP, TN, and FN for each class
  * For class balanced datasets:
    * Accuracy : (TP+TN)/(TP+FP+TN+FN)
    * ROC curve: TPR vs FPR
  * For class imbalanced datasets:
    * Precision (PR): TP/(TP+FP)
    * Recall (RC): TP/(TP+FN)
    * F1-Score: 2*PR*RC/(PR+RC)
    * Balanced accuracy: (TPR+TNR)/2
    * Weighted-Averaged Precision, Recall, and F1-Score
    * PR curve: PR vs RC
* Performance - Detection
  * Intersection over Union (IoU)
  * mAP: Average AP over all classes
  * mAP@0.5: Uses IoU threshold 0.5 (PASCAL VOC)
  * mAP@0.5:0.95: Averages AP over multiple IoU thresholds (COCO metric)
  * False Positives Per Image (FPPI)
  * Precision, Recall, and F1-Score
* Performance - Segementation
  * Intersection over Union (IoU) / Jaccard Index
  * Dice Coefficient / F1-Score
  * Mean Pixel Accuracy (mPA)
  * Boundary IoU (BIoU)
  * Hausdorff Distance
  * Precision
  * Recall / Sensitivity / True Positive Rate
* Performance - Tracking
  * Multiple Object Tracking Accuracy (MOTA)
  * Multiple Object Tracking Precision (MOTP)
  * ID F1-Score (IDF1)
  * Identity Switches (IDSW)
  * Track Completeness (TC)
  * Mostly Tracked (MT) / Mostly Lost (ML)
* Performance - Perceptual Quality (Super-resolution, Denoising, Contrast Enhancement)
  * Peak Signal-to-Noise Ratio (PSNR)
  * Mean Squared Error (MSE)
  * Structural Similarity Index (SSIM)
  * Multi-Scale SSIM (MS-SSIM)
  * Learned Perceptual Image Patch Similarity (LPIPS)
  * Visual Information Fidelity (VIF)
  * Kernel Inception Distance (KID)
  * Gradient Magnitude Similarity Deviation (GMSD)
  * Edge Preservation Index (EPI)
  * Natural Image Quality Evaluator (NIQE)
* Performance - Generation (GANs, Diffusion Models)
  * Inception Score (IS)
  * Fréchet Inception Distance (FID)
  * Perceptual Path Length (PPL)
* Computation
  * Inference Time - Frames Per Second (FPS)
  * Model Size

---

## Conferences

* CORE Rank A:
  * ICCV: International Conference on Computer Vision (IEEE) [[dblp](https://dblp.org/streams/conf/iccv)]
  * CVPR: Conference on Computer Vision and Pattern Recognition (IEEE) [[dblp](https://dblp.org/streams/conf/cvpr)]
  * ECCV: European Conference on Computer Vision (Springer) [[dblp](https://dblp.org/streams/conf/eccv)]
  * WACV: Winter Conference/Workshop on Applications of Computer Vision (IEEE) [[dblp](https://dblp.org/streams/conf/wacv)]
  * ICASSP: International Conference on Acoustics, Speech, and Signal Processing (IEEE) [[dblp](https://dblp.org/streams/conf/icassp)]
  * MICCAI: Conference on Medical Image Computing and Computer Assisted Intervention (Springer) [[dblp](https://dblp.org/streams/conf/miccai)]
  * ISBI: IEEE International Symposium on Biomedical Imaging (IEEE) [[dblp](https://dblp.org/streams/conf/isbi)]
  * IROS: International Conference on Intelligent Robots and Systems (IEEE) [[dblp](https://dblp.org/streams/conf/iros)]
  * ACMMM: ACM International Conference on Multimedia (ACM) [[dblp](https://dblp.org/streams/conf/mm)]
* CORE Rank B
  * ACCV: Asian Conference on Computer Vision (Springer) [[dblp](https://dblp.org/streams/conf/accv)]
  * VCIP: International Conference on Visual Communications and Image Processing (IEEE) [[dblp](https://dblp.org/streams/conf/vcip)]
  * ICIP: International Conference on Image Processing (IEEE) [[dblp](https://dblp.org/streams/conf/icip)]
  * CAIP: International Conference on Computer Analysis of Images and Patterns (Springer) [[dblp](https://dblp.org/streams/conf/caip)]
  * VISAPP: International Conference on Vision Theory and Applications (SCITEPRESS) [[dblp](https://dblp.org/streams/conf/visapp)]
  * ICPR: International Conference on Pattern Recognition (IEEE) [[dblp](https://dblp.org/streams/conf/icpr)]
  * ACIVS: Conference on Advanced Concepts for Intelligent Vision Systems (Springer) [[dblp](https://dblp.org/streams/conf/acivs)]
  * EUSIPCO: European Signal Processing Conference (IEEE) [[dblp](https://dblp.org/streams/conf/eusipco)]
  * ICRA: International Conference on Robotics and Automation (IEEE) [[dblp](https://dblp.org/streams/conf/icra)]
  * BMVC: British Machine Vision Conference (organized by BMVA: British Machine Vision Association and Society for Pattern Recognition) [[dblp](https://dblp.org/streams/conf/bmvc)]
* CORE Rank C:
  * ICISP: International Conference on Image and Signal Processing (Springer) [[dblp](https://dblp.org/streams/conf/icisp)]
  * ICIAR: International Conference on Image Analysis and Recognition (Springer) [[dblp](https://dblp.org/streams/conf/iciar)]
  * ICVS: International Conference on Computer Vision Systems (Springer) [[dblp](https://dblp.org/streams/conf/icvs)]
* Unranked but popular
  * MIUA: Conference on Medical Image Understanding and Analysis (organized by BMVA: British Machine Vision Association and Society for Pattern Recognition) [[dblp](https://dblp.org/streams/conf/miua)]
  * EUVIP: European Workshop on Visual Information Processing (IEEE, organized by EURASIP: European Association for Signal Processing) [[dblp](https://dblp.org/streams/conf/euvip)]
  * CIC: Color and Imaging Conference (organized by IS&T: Society for Imaging Science and Technology) [[dblp](https://dblp.org/streams/conf/imaging)]
  * CVCS: Colour and Visual Computing Symposium [[dblp](https://dblp.org/streams/conf/cvcs)]
  * DSP: International Conference on Digital Signal Processing [[dblp](https://dblp.org/streams/conf/icdsp)]

---

## Journals

* Tier 1
  * IEEE Transactions on Pattern Analysis and Machine Intelligence (IEEE TPAMI) [[dblp](https://dblp.org/streams/journals/pami)]
  * Springer International Journal of Computer Vision (Springer IJCV) [[dblp](https://dblp.org/streams/journals/ijcv)]
  * IEEE Transactions on Image Processing (IEEE TIP) [[dblp](https://dblp.org/streams/journals/tip)]
  * IEEE Transactions on Circuits and Systems for Video Technology (IEEE TCSVT) [[dblp](https://dblp.org/streams/journals/tcsv)]
  * Elsevier Pattern Recognition (Elsevier PR) [[dblp](https://dblp.org/streams/journals/pr)]
  * Elsevier Computer Vision and Image Understanding (Elsevier CVIU) [[dblp](https://dblp.org/streams/journals/cviu)]
  * Elsevier Expert Systems with Applications [[dblp](https://dblp.org/streams/journals/eswa)]
  * Elsevier Neurocomputing [[dblp](https://dblp.org/streams/journals/ijon)]
  * Springer Neural Computing and Applications [[dblp](https://dblp.org/streams/journals/nca)]
  * IEEE Transactions on Medical Imaging (IEEE TMI) [[dblp](https://dblp.org/streams/journals/tmi)]
  * Elsevier Medical Image Analysis [[dblp](https://dblp.org/streams/journals/mia)]
  * Elsevier Computerized Medical Imaging and Graphics [[dblp](https://dblp.org/streams/journals/cmig)]
  * Elsevier Computer Methods and Programs in Biomedicine [[dblp](https://dblp.org/streams/journals/cmpb)]
  * Elsevier Computers in Biology and Medicine [[dblp](https://dblp.org/streams/journals/cbm)]
* Tier 2
  * Elsevier Image and Vision Computing (Elsevier IVC) [[dblp](https://dblp.org/streams/journals/ivc)]
  * Elsevier Pattern Recognition Letters (Elsevier PR Letters) [[dblp](https://dblp.org/streams/journals/prl)]
  * Elsevier Journal of Visual Communication and Image Representation [[dblp](https://dblp.org/streams/journals/jvcir)]
  * Springer Journal of Mathematical Imaging and Vision [[dblp](https://dblp.org/streams/journals/jmiv)]
  * SPIE Journal of Electronic Imaging [[dblp](https://dblp.org/streams/journals/jei)]
  * IET Image Processing [[dblp](https://dblp.org/streams/journals/iet-ipr)]
  * Springer Pattern Analysis and Applications (Springer PAA) [[dblp](https://dblp.org/streams/journals/paa)]
  * Springer Machine Vision and Applications (Springer MVA) [[dblp](https://dblp.org/streams/journals/mva)]
  * IET Computer Vision [[dblp](https://dblp.org/streams/journals/iet-cvi)]
* Open Access
  * IEEE Access [[dblp](https://dblp.org/streams/journals/access)]
  * MDPI Journal of Imaging [[dblp](https://dblp.org/streams/journals/jimaging)]

---

## Summer Schools

* International Computer Vision Summer School (IVCSS) [2007-Present], Sicily, Italy [[Website](https://iplab.dmi.unict.it/icvss2023/)]
* Machine Intelligence and Visual Computing Summer School (VISUM) [2013-2022], Porto, Portugal [[Website](https://visum.inesctec.pt)]
* BMVA British Computer Vision Summer School (CVSS) [2013-2020,2023-Present], UK [[Website](https://britishmachinevisionassociation.github.io/summer-school)]

---

## Popular Articles

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
* Image Generation
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
  * Explainable AI (XAI)
  * Video Summarization and Captioning
  * Text Recognition
  * Data Compression
  * Affective Computing
  * Image Colorization
  * Virtual reality (VR)
  * Augmented reality (AR)
  * Visual Question Answering (VQA)
  * Vision-Language Models (VLMs)
  * DeepFake Detection
  * 3D Reconstruction
  * Image Captioning
  * Image Super-Resolution / Image Restoration
  * Pose Estimation
  * Biometric Analysis
  * Depth Estimation
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
| [Introduction to Computer Vision](https://browncsci1430.github.io/webpage/) | 2025 | James Tompkin | Brown |
| [Deep Learning for Computer Vision](http://cs231n.stanford.edu) | 2024 | Fei-Fei Li | Stanford |
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

* Tags: Object Classification `[ObjCls]`, Object Detection `[ObjDet]`, Object Segmentation `[ObjSeg]`, General Library `[GenLib]`, Text Reading / Object Character Recognition `[OCR]`, Action Recognition `[ActRec]`, Object Tracking `[ObjTrk]`, Data Augmentation `[DatAug]`, Simultaneous Localization and Mapping `[SLAM]`, Outlier/Anomaly/Novelty Detection `[NvlDet]`, Content-based Image Retrieval `[CBIR]`, Image Enhancement `[ImgEnh]`, Aesthetic Assessment `[AesAss]`, Explainable Artificial Intelligence `[XAI]`, Text-to-Image Generation `[TexImg]`, Pose Estimation `[PosEst]`, Video Matting `[VidMat]`, Eye Tracking `[EyeTrk]`

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

* [@AurelienGeron](https://www.youtube.com/@AurelienGeron) `[Individual]`, Aurélien Géron: former lead of YouTube's video classification team, and author of the O'Reilly book Hands-On Machine Learning with Scikit-Learn and TensorFlow.
* [@howardjeremyp](https://www.youtube.com/@howardjeremyp) `[Individual]`, Jeremy Howard: former president and chief scientist of Kaggle, and co-founder of fast.ai.
* [@PieterAbbeel](https://www.youtube.com/@PieterAbbeel) `[Individual]`, Pieter Abbeel: professor of electrical engineering and computer sciences, University of California, Berkeley.
* [@pascalpoupart3507](https://www.youtube.com/@pascalpoupart3507) `[Individual]`, Pascal Poupart: professor in the David R. Cheriton School of Computer Science at the University of Waterloo.
* [@MatthiasNiessner](https://www.youtube.com/@MatthiasNiessner) `[Individual]`, Matthias Niessner: Professor at the Technical University of Munich and head of the Visual Computing Lab.
* [@MichaelBronsteinGDL](https://www.youtube.com/@MichaelBronsteinGDL) `[Individual]`, Michael Bronstein: DeepMind Professor of AI, University of Oxford / Head of Graph Learning Research, Twitter.
* [@DeepFindr](https://www.youtube.com/@DeepFindr) `[Individual]`, Videos about all kinds of Machine Learning / Data Science topics.
* [@deeplizard](https://www.youtube.com/@deeplizard) `[Individual]`, Videos about building collective intelligence.
* [@YannicKilcher](https://www.youtube.com/@YannicKilcher) `[Individual]`, Yannic Kilcher: make videos about machine learning research papers, programming, and issues of the AI community, and the broader impact of AI in society.
* [@sentdex](https://www.youtube.com/@sentdex) `[Individual]`, sentdex: provides Python programming tutorials in machine learning, finance, data analysis, robotics, web development, game development and more.
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
* WIP:
  * <https://www.youtube.com/@AAmini>
  * <https://www.youtube.com/@WhatsAI>  
  * <https://www.youtube.com/@mrdbourke>
  * <https://www.youtube.com/@marksaroufim>
  * <https://www.youtube.com/@NicholasRenotte>
  * <https://www.youtube.com/@abhishekkrthakur>
  * <https://www.youtube.com/@AladdinPersson>
  * <https://www.youtube.com/@CodeEmporium>
  * <https://www.youtube.com/@arp_ai>
  * <https://www.youtube.com/@CodeThisCodeThat>
  * <https://www.youtube.com/@connorshorten6311>
  * <https://www.youtube.com/@SmithaKolan>
  * <https://www.youtube.com/@AICoffeeBreak>
  * <https://www.youtube.com/@independentcode>
  * <https://www.youtube.com/@alfcnz>
  * <https://www.youtube.com/@KapilSachdeva>
  * <https://www.youtube.com/@AICoding>
  * <https://www.youtube.com/@mildlyoverfitted>

---

## Mailing Lists

* [Vision Science](http://visionscience.com/mailman/listinfo/visionlist_visionscience.com), announcements about industry/academic jobs in computer vision around the world (in English).
* [bull-i3](https://listes.irit.fr/sympa/info/bull-i3), posts about job opportunities in computer vision in France (in French).

---

## Misc

* How to build a good poster - [[Link1](https://urc.ucdavis.edu/sites/g/files/dgvnsk3561/files/local_resources/documents/pdf_documents/How_To_Make_an_Effective_Poster2.pdf)] [[Link2](https://www.animateyour.science/post/How-to-design-an-award-winning-conference-poster)] [[Link3](https://www.jamiebgall.co.uk/post/powerful-posters)]
* How to report a good report - [[Link1](https://cs.swan.ac.uk/~csbob/teaching/cs354-projectSpec/laramee10projectGuideline.pdf)] [[link2](https://www.cst.cam.ac.uk/teaching/part-ii/projects/dissertation)]
* [The "Python Machine Learning (3rd edition)" book code repository](https://github.com/rasbt/python-machine-learning-book-3rd-edition)
* [Multithreading with OpenCV-Python to improve video processing performance](https://nrsyed.com/2018/07/05/multithreading-with-opencv-python-to-improve-video-processing-performance/)
* [Computer Vision Zone](https://www.computervision.zone/) - Videos and implementations for computer vision projects
* [MadeWithML](https://github.com/GokuMohandas/MadeWithML), Learn how to responsibly deliver value with ML
* [d2l-en](https://github.com/d2l-ai/d2l-en), Interactive deep learning book with multi-framework code, math, and discussions. Adopted at 200 universities
* [Writing Pet Peeves](https://www.cs.ubc.ca/~tmm/writing.htmt), writing guide for correctness, references, and style
* [Hitchhiker's Guide to Python](https://docs.python-guide.org), Python best practices guidebook, written for humans
* [python-fire](https://github.com/google/python-fire), Google, a library for automatically generating command line interfaces (CLIs) from absolutely any Python object.
* [shotcut](https://shotcut.org), a free, open source, cross-platform video editor.
* [PyTorch Computer Vision Cookbook](https://github.com/PacktPublishing/PyTorch-Computer-Vision-Cookbook), PyTorch Computer Vision Cookbook, Published by Packt.
* [Machine Learning Mastery - Blogs](https://machinelearningmastery.com/blog/), Blogs written by [Jason Brownlee](https://scholar.google.com/citations?hl=en&user=hVaJhRYAAAAJ) about machine learning.
* [PyImageSearch - Blogs](https://pyimagesearch.com/blog/), Blogs written by [Adrian Rosebrock](https://scholar.google.com/citations?user=bLEhONMAAAAJ&hl) about computer vision.
* [jetson-inference](https://github.com/dusty-nv/jetson-inference), guide to deploying deep-learning inference networks and deep vision primitives with TensorRT and NVIDIA Jetson.

---

## Thanks

* [Frida de Sigley](https://github.com/fdsig)
* Dan Harvey
* [CORE Conference Ranking](http://portal.core.edu.au/conf-ranks/?search=4603&by=all&source=CORE2021&sort=arank&page=1)
* [Scimago Journal Ranking](https://www.scimagojr.com/journalrank.php)
* [benthecoder/yt-channels-DS-AI-ML-CS](https://github.com/benthecoder/yt-channels-DS-AI-ML-CS)
* [anomaly-detection-resources](https://github.com/yzhao062/anomaly-detection-resources), Anomaly detection related books, papers, videos, and toolboxes
* [awesome-satellite-imagery-datasets](https://github.com/chrieke/awesome-satellite-imagery-datasets) List of satellite image training datasets with annotations for computer vision and deep learning
* [awesome-Face_Recognition](https://github.com/ChanChiChoi/awesome-Face_Recognition), Computer vision papers about faces.
* [the-incredible-pytorch](https://github.com/ritchieng/the-incredible-pytorch), Curated list of tutorials, papers, projects, communities and more relating to PyTorch
