# DeepID 2025: The Challenge Of Detecting Synthetic Manipulations In ID Documents

<p align='center'>  
  <img src='imgs/title2.jpg' />
</p>

Both Tracks’ 1st Place Solution for [**The Challenge of Detecting Synthetic Manipulations in ID Documents (Detection  Track & Localization Track)**](https://deepid-iccv.github.io/) by **"Sunlight"** team.

Team members: **[Zeqin Yu](https://zeqinyu.github.io/aboutme/)**<sup>1</sup>, **Ye Tian**<sup>2</sup>, under the supervision of **[Prof. Jiangqun Ni](https://scst.sysu.edu.cn/members/members01/1408534.htm)**<sup>2,3</sup>.

<sup>1</sup> *School of Computer Science and Engineering, Sun Yat-sen University*  
<sup>2</sup> *School of Cyber Science and Engineering, Sun Yat-sen University*  
<sup>3</sup> *Peng Cheng Laboratory*


## Table of Contents

- [Our Solution](#our-solution)
  - [Preliminary Analysis](#preliminary-analysis)
  - [Proposed Pipeline](#proposed-pipeline)
- [Competition Results](#competition-results)
- [Acknowledgement](#acknowledgement)
  


## Our Solution
We proposed a two-stage training pipeline based on the **[Reinforced Multi-teacher Knowledge Distillation (Re-MTKD)](https://ojs.aaai.org/index.php/AAAI/article/view/32085)**  framework.

<p align="center">
  <img src="imgs/frame.jpg" alt="frame" width="800"/>
  <br/>
  <em>Figure 1. Overview of our two-stage training pipeline.</em>
</p>


### Preliminary Analysis
**Analysis1:** We conducted an initial analysis to understand the domain-specific characteristics of the ID document images in the challenge. We observed that real and tampered images differ in JPEG compression quality (QF95 vs. QF75), and that the images feature structured layouts, clean backgrounds, and concentrated text regions. Such properties pose challenges for general-purpose image forgery detection and localization (IFDL) models, particularly in modeling compression artifacts and document-specific textures.

**Analysis2:** We also evaluated the zero-shot performance of several state-of-the-art IFDL methods, including MVSS-Net, TruFor (official baseline), and our own **[Re-MTKD](https://ojs.aaai.org/index.php/AAAI/article/view/32085)** . As shown in Tab.1, Re-MTKD achieved the highest F1 scores in both detection and localization despite being trained on fewer samples, indicating strong cross-domain generalization.

<div align="center">

<em> Table 1. Zero-shot performance (F1 score) of existing IFDL methods on the FantasyID validation set.</em>

| Method    | Data  | Detection | Localization | Average |
|:----------:|:-----:|:---------:|:------------:|:-------:|
| MVSS-Net  | 13k   | 0.533     | 0.187        | 0.360   |
| TruFor    | 876k  | 0.746     | 0.626        | 0.686   |
| Re-MTKD   | 60k   | 0.758     | 0.637        | 0.697   |


</div>

### Proposed Pipeline
In Stage 1 (left part of Fig. 1), we reused the Cue-Net student model pretrained using the Re-MTKD framework, which has shown strong generalization across various natural image forgery datasets. We trained teacher models, each focusing on a specific manipulation type (e.g., splicing, copy-move, inpainting), on datasets corresponding to each manipulation type, such as CASIAv2  for copy-move, FantasticReality for splicing, and GC for inpainting. We optimized the student model using a combination of soft distillation loss Lsoft and hard supervision loss Lhard, which combines segmentation, classification, and edge-aware losses to enhance detection accuracy and localization precision.

In Stage 2 (right part of Fig. 1), we fine-tuned the pretrained model on the FantasyID dataset to capture ID-specific characteristics such as structured layouts, subtle tampering traces, and compression-induced artifacts. We conducted training on 512×512 cropped patches with randomized JPEG compression (QF in [70, 100]) applied to simulate the diverse compression artifacts observed in the provided FantasyID dataset. During fine-tuning, we only optimized the hard supervision loss Lhard to ensure stable domain adaptation in the absence of the teacher’s guidance. At inference, we used whole-image processing to avoid resizing artifacts. This approach, when trained for 16 epochs, achieved first place in the detection track, and when trained for 31 epochs, first place in the localization track.

## Competition Results
The F1 scores ranking the winning teams (see Tab. 2) are reported in Tab. 3 for the detection track and in Tab. 4 for the localization track. The columns t₍f₎ and F1₍f₎ denote the inference time and F1 score on the FantasyID set, while t₍p₎ and F1₍p₎ correspond to the results on the private set.

<p align="center">
  <em> Table 2. The teams whose submissions scored above the TruFor baseline.</em>
  <br/>
  <img src="imgs/tab1.jpg" alt="frame" width="500"/>
  

</p>

<p align="center">
  <em> Table 3. Detection performance of winning teams in DeepID.</em>
   <br/>
  <img src="imgs/tab2.jpg" alt="frame" width="500"/>
 

</p>

<p align="center">
  <em> Table 4. Localization performance of winning teams in DeepID.</em>
  <br/>
  <img src="imgs/tab3.jpg" alt="frame" width="500"/>
  

</p>

Our team (**Sunlight**) achieves the fastest inference speed while performing both detection and localization. It ranks first on FantasyID and remains highly competitive on the private set. Overall, **Sunlight** secures the top position in the aggregated score across both tracks, demonstrating the superiority of our approach.


## Acknowledgement



<!-- ## 📊 Final Scoring (Aggregate F1 Score)

Final ranking uses a **weighted F1 score** formula:

$$
\text{Aggregate\_F1} = 0.3 \times \text{F1}_{\text{fantasy}} + 0.7 \times \text{F1}_{\text{private}}
$$



Where:

- `F1_fantasy` = average of F1 on bonafide and attack samples from the Fantasy test set  
- `F1_private` = F1-score on the private real document test set   -->










