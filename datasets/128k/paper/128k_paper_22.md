<paper 0>
# Generative AI for Integrated Sensing and Communication: Insights from the Physical Layer Perspective 

Jiacheng Wang, Hongyang Du, Dusit Niyato, Fellow, IEEE, Jiawen Kang, Shuguang Cui, Fellow, IEEE,<br>Xuemin (Sherman) Shen, Fellow, IEEE, and Ping Zhang, Fellow, IEEE


#### Abstract

As generative artificial intelligence (GAI) models continue to evolve, their generative capabilities are increasingly enhanced and being used extensively in content generation. Beyond this, GAI also excels in data modeling and analysis, benefitting wireless communication systems. In this article, we investigate applications of GAI in the physical layer and analyze its support for integrated sensing and communications (ISAC) systems. Specifically, we first provide an overview of GAI and ISAC, touching on GAI's potential support across multiple layers of ISAC. We then concentrate on the physical layer, investigating GAI's applications from various perspectives thoroughly, such as channel estimation, and demonstrate the value of these GAIenhanced physical layer technologies for ISAC systems. In the case study, the proposed diffusion model-based method effectively estimates the signal direction of arrival under the near-field condition based on the uniform linear array, when antenna spacing surpassing half the wavelength. With a mean square error of 1.03 degrees, it confirms GAI's support for the physical layer in near-field sensing and communications.


Index Terms-Generative AI, integrated sensing and communications, physical layer, diffusion model

## I. INTRODUCTION

Recently, the unprecedented growth in user data, together with the continuous advancement of AI-Generated Content (AIGC) models, have led to groundbreaking applications such as Google Bard and ChatGPT. As users increasingly benefit from these applications, their attention is concurrently shifting to the mechanism powering these applications, i.e., generative artificial intelligence (GAI) [1]. Unlike traditional AI models that prioritize sample analysis, training, and classification, GAI specializes in understanding and modeling the distribution of complex datasets. By leveraging statistical properties of the training data, GAI can generate data similar to the training data, manifesting in diverse formats like documents and images [2]. For example, the diffusion model-based ControlNet [3] can efficiently generate images with outstanding

J. Wang, H. Du and D. Niyato are with the School of Computer Science and Engineering, Nanyang Technological University, Singapore (e-mail: jiacheng.wang@ntu.edu.sg, hongyang001@e.ntu.edu.sg, dniyato @ ntu.edu.sg).

J. Kang is with the School of Automation, Guangdong University of Technology, Guangzhou, China (e-mail: kavinkang @ gdut.edu.cn).

S. Cui is with the School of Science and Engineering (SSE) and the Future Network of Intelligence Institute (FNii), Chinese University of Hong Kong (Shenzhen), China (e-mail: shuguangcui@cuhk.edu.cn).

X. Shen is with the Department of Electrical and Computer Engineering, University of Waterloo, Canada (email: sshen@uwaterloo.ca).

P. Zhang is with the State Key Laboratory of Networking and Switching Technology, Beijing University of Posts and Telecommunications, China (email: pzhang @bupt.edu.cn). quality, in terms of resolution, saturation, and naturalness, according to the user prompts, which demonstrates greater flexibility and higher efficiency compared to traditional content generation methods. In the context of the rapidly evolving of wireless network services, GAI is poised to meet the various and ever-changing demands of users.

Indeed, not only content generation, GAI's inference capability has catalyzed research across various domains. For example, researchers in [4] introduce a generative adversarial networks (GANs) based architecture to learn the channel transition probabilities (CTP) from the observations, thereby helping achieve the maximum likelihood sequence detection. Additionally, in device-to-device (D2D) communications, an incentive mechanism based on contract theory is proposed to facilitate user information sharing, in which the diffusion model is employed to generate optimal contract designs [2].

While there have been attempts to integrate GAI into wireless communication, they remain limited, especially when considering emerging technologies like extremely largescale multiple-input-multiple-output (XL-MIMO), near-field communications, and integrated sensing and communication (ISAC) [5]. For instance, ISAC encompasses both communication and sensing modules, as shown in Fig. 1, and each module has specific demands for bandwidth, power, and other resources. This complexity imposes new challenges in designing efficient wireless resource allocation strategies at the network layer to balance sensing and communication.

Moreover, physical layer technologies such as antenna array and waveform design are also crucial for ISAC systems. For communication purposes, enhancing transmission reliability in multipath fading channels necessitates large antenna spacing to ensure independent signals across antennas. On the other hand, for sensing, estimating key parameters including the direction of arrival (DoA) of signal waves usually require antenna spacing to be less than or equal to half the wavelength to avoid ambiguities. These conflicting requirements introduce new challenges in the design of the antenna array for ISAC systems. Fortunately, the emerging of GAI and its recent applications in wireless communications, provide a promising way for resolving these dilemmas. Therefore, an in-depth investigation into the applications of GAI in ISAC systems, particularly in the physical layer, is imperative.

Recognizing the challenges outlined above, this article conducts an extensive investigation on the application of GAI in the physical layer and the corresponding potential support

![](https://cdn.mathpix.com/cropped/2024_06_04_c7e6d93cb290df792351g-2.jpg?height=1272&width=1792&top_left_y=188&top_left_x=161)

Fig. 1. The role of GAI in the physical layer and its support for ISAC applications. The GAI models can be utilized to enhance several physical layer technologies, including channel state information (CSI) compression and signal detection. On this basis, the GAI enhanced physical layer technologies can further augment ISAC system performance across various applications, such as indoor human detection and outdoor vehicle to vehicle communication.

for ISAC systems. Concretely, we first present an overview of five major GAI models and the ISAC system. After that, we thoroughly analyze the potential support of these GAIenhanced physical layer technologies for ISAC from both sensing and communication perspectives. Finally, we provide a practical use case to explain how GAI can be used to tackle challenges in signal DoA estimation during sensing, a critical component of ISAC. Overall, the contributions of this article are summarized as follows.

- We conduct a review of five major GAI models and the ISAC system. Building on this, we analyze the potential applications of the GAI models in the physical, network, and application layers of the ISAC system, providing comprehensive insights of emerging sensing, localization, and communication technologies.
- From different perspectives such as beamforming and signal detection, we investigate how GAI models enhance physical layer technologies. Subsequently, we analyze the support that these GAI-enhanced physical layer technologies provide for communication and sensing in ISAC systems, outlining technical issues and viable solutions.
- We propose a signal spectrum generator (SSG) to tackle the near-field DoA estimation problem when antenna spacing exceeds half the wavelength. Experimental results reveal that SSG yields a mean square error (MSE) of around 1.03 degrees in DoA estimation, confirming SSG's effectiveness while highlighting the importance of integrating GAI into the ISAC physical layer.


## II. OVERVIEW OF GENERATIVE AI AND ISAC

This section first introduces the concepts of GAI and presents five representative GAI models. Following that, we introduce ISAC and generally explain GAI's potential support for ISAC systems from the physical, network, and application layers.

## A. Generative AI

GAI refers to a specific category of AI models that is trained on large datasets to learn the inherent patterns of the data distribution. Once trained, they can generate new data that is similar yet distinct from the training data, facilitating content production. Compared to the traditional AI models, the GAI models hold better ability to understand and capture the distribution of the complex and high-dimensional data [1]. Hence,
they find several applications across various fields. Among different GAI models, GANs, normalizing flows (NFs), variational autoencoders (VAEs), diffusion models (DFMs), and Transformers not only excel in generating digital content but also demonstrate significant applicability in the physical layer of wireless communications.

- GANs (Fig. 1-part A) consist of a generator and a discriminator that compete during training, aiming for a particular equilibrium. The training is completed when the discriminator cannot differentiate between real and fake data. After that, the generator can produce similar, yet new data in a parallel manner. However, the training process is complex, as finding the equilibrium is harder than optimizing an objective function.
- NFs (Fig. 1-part B) use invertible transformations to map basic distributions to target spaces for detailed analysis. These transformations create a flow that can be reversed, facilitating likelihood estimation. NFs can sample from complex probability distributions, which is useful for the unanalyzable data. However, many transformations may make the training process time-consuming.
- VAEs (Fig. 17part C) are neural networks designed to compress and reconstruct data. Unlike traditional autoencoders, VAEs can model the latent distribution and sample from the modeled space, benefiting data dimension reduction and feature extraction. Additionally, they can estimate the uncertainty in predictions and generate plausible outputs for a given input. However, generated samples are not always interpretable, as they are derived from the latent space.
- DFMs (Fig. 1-part D) have attracted significant attention due to their image generation capabilities. During the training, DFMs corrupt training data with random noise and subsequently denoise the data to learn optimal hyperparameters. Once trained, they apply the learned parameters in the denoising process to generate samples. DFMs can be trained on incomplete datasets with a stable process, but inference requires many steps, making them less efficient for generating large datasets.
- Transformers (Fig. 1-part E) are neural network architectures based on the self-attention mechanism, which can model long-range dependencies between elements in the input sequence and support parallel sequences processing, suitable for tasks involving substantial sequence data. Their design needs minimal inductive biases and is inherently suited for set-functions, enabling them to process multiple modalities using similar processing blocks.

Besides the above-mentioned models, there are some other GAI models, such as multimodal models, used in ChatGPT. These models possess strong data analysis and modeling capabilities, making them advantageous for incorporation into communication system designs.

## B. Integrated Sensing and Communication

ISAC focuses on integrating wireless sensing and communication into a unified system. This aims at the efficient use of limited resources, while facilitating both functions [5].
From the physical layer, ISAC can be broadly classified into non-overlapping and overlapping systems. Specifically, nonoverlapping systems include time-division, frequency-division, and space-division ISAC. For example, time-division ISAC allocates distinct signals to individual time slots for either sensing or communication tasks, allowing them to use their preferred waveforms. The overlapping systems can be divided into sensing-centric, communication-centric, and joint designs. For example, the communication-centric design can be achieved by appropriately modifying existing communication systems, and a representative example is Wi-Fi sensing. Compared to traditional wireless communication and sensing systems, the ISAC systems offer several advantages.

- Higher efficiency: By allowing communication and sensing to share resources, ISAC boosts the overall efficiency of wireless networks.
- Lower cost: By eliminating the need for separate communication and sensing modules, ISAC lowers both hardware and power consumption costs for wireless devices.
- More versatile services: ISAC is capable of fulfilling users' communication requirements while concurrently offering sensing function, allowing it to deliver more services.

Benefiting from these advantages, ISAC systems can be applied across various scenarios and are thus considered one of the core technologies for future $6 \mathrm{G}$ networks.

## C. Potential Applications of GAI in ISAC Systems

As aforementioned, we can see that GAI can serve ISAC systems from multiple perspectives. This can be broadly categorized into the physical, network, and application layers.

- Physical layer: GAI can be employed for channel estimation, anomaly signal identification, encoding, beamforming, etc, as shown in Fig. 1. These GAI-enhanced physical layer technologies can improve the communication performance (e.g., reducing bit error rate (BER)) and enhancing the sensing accuracy (e.g., optimizing signal beams to increase target detection accuracy while avoiding interference in ISAC systems).
- Network layer: GAI can be utilized for designing resource allocation strategies, scheduling schemes, and incentive mechanisms, which could not only lower the system cost but also boost the operation efficiency. While methods such as deep reinforcement learning (DRL) are applicable here, GAI has been shown to be more effective in tasks like resource allocation [2].
- Application layer: GAI can be used to offer support in data generation, analysis, and feature extraction for various ISAC applications. This support not only facilitates in-depth analysis of communication or sensing data but also generates a substantial amount of data for both communication and sensing model training, which is difficult for other existing AI models.

In Table I, we summarize the above mentioned GAI models and their potential support for ISAC systems. Next, we detail GAI's applications in the physical layer.

TABLE I

FIVE TYPICAL GAI MODELS AND CORRESPONDING POTENTIAL SUPPORT FOR ISAC AT DIFFERENT LAYERS.

![](https://cdn.mathpix.com/cropped/2024_06_04_c7e6d93cb290df792351g-4.jpg?height=908&width=1808&top_left_y=275&top_left_x=148)

## III. GAI-ENHANCED PHYSICAL LAYER TECHNOLOGIES FOR ISAC

The physical layer includes several key technologies such as codebook design and channel estimation. In this section, we investigate how GAI strengthens various physical layer technologies and discuss their potential support for ISAC systems from both sensing and communication perspectives.

## A. From Communication Perspective

1) Signal Detection: Detecting signals in cases with unpredictable noise is challenging. NFs can infer latent variables, offering an effective solution. Hence, the authors in [6] propose a probabilistic machine-learning detection framework that employs NFs to approximate the unknown system noise in MIMO systems without any prior information. This approximation is driven by unsupervised learning with only noise samples, which is difficult to achieve with traditional AI models. Evaluations show that this framework not only stands out in terms of BER in environments with unanalyzable noise, but also reaches close to the maximum likelihood bound in environments with predictable noise. Besides NFs, other GAI models like GANs and VAEs can be also used for signal detection. In ISAC systems, the integration of communication and sensing creates more complex noise, additionally, differences in signal waveforms and other aspects between these two modules could exacerbate the issue. Therefore, NFs can also be employed to model the unknown noise, improving signal detection capability of ISAC systems.
2) Secure Transceiver Design: The complexity of ISAC architectures and channel models complicates the design of security technologies. With the ability of processing complex data, VAEs can automatically manage codeword variation, which can be modeled as noise during transmission, making VAEs suitable for building secure transceiver pairs. In [7], the authors modify the VAE loss function at the receiver to include a security term, enhancing the receiver security. The unsupervised training is further used to strengthen the robustness against random codeword variations. In the case of imperfect CSI with the signal-to-noise ratio (SNR) range from $-5 \mathrm{~dB}$ to $10 \mathrm{~dB}$, the BER of this method at the eavesdropper is 0.05 higher than that of the autoencoder based on traditional neural networks. The same approach can be integrated into ISAC systems to enhance the security of the receiver and the robustness to codeword variations. However, when sensing and communication share the receiver, it is crucial to consider how adding the security term to a loss function might affect the sensing module.
3) Sparse Code Multiple Access: In ISAC, various smart devices like unmanned aerial vehicles participate in communication and sensing, causing severe interference among devices. To mitigate this, combining GAI models with non-orthogonal multiple access (NOMA) techniques is a promising solution. The authors in [8] introduce a GAN-based sparse code multiple access (SCMA) encoding and decoding approach. At the SCMA encoder, the generator is used to shorten the sequences in the information processing phase. Additionally, a noise layer is introduced to ensure a robust representation of the encoder output, thereby improving the noise immunity. At the decoder, PatchGAN serves as the discriminator to reduce both model parameters and computational load. Besides, an attention mechanism is inserted between the GAN's generator and discriminator to enhance the BER performance. Such designs can offer better connectivity of various smart devices involved in communication for ISAC, ensuring that control,
scheduling, and other information can be timely transmitted to each device.
4) Joint Source-Channel Coding: Coding is crucial for mitigating channel noise and interference, making it essential for communication of ISAC. Joint source-channel coding (JSCC) is an effective encoding method, but the complexity and discontinuity of the source data distribution present design challenges. To address this, in [9], the authors employ the VAE encoder to transform source data into a low-dimensional latent space and use the decoder to revert it to the original data for JSCC. During this process, one of multiple encoders is selected for transmission to tackle the issue of discontinuous projection. The evaluations show that the average peak SNR (PSNR) of the proposed method is nearly $3 \mathrm{~dB}$ higher than traditional methods based on convolutional neural networks. In ISAC systems where communication and sensing modules have independent encoding requirements and the channel is modeled as an additive Gaussian noise channel, such a method can directly contribute to the JSCC efficiency of communication module in ISAC.

## B. From Sensing Perspective

1) CSI Compression: Sensing in ISAC may need a significant amount of CSI from multiple antennas and frequencies, especially in systems like Wi-Fi based sensing. Hence, efficient compression, which facilitates the CSI storage and transmission, is essential. Given the superiority over traditional multi-layer perceptrons when output dimensionality far exceeds input, GANs are a preferred choice for CSI compression. In [10], the authors use the CSiNet encoder at the transmitter to compress original CSI into a low-dimensional vector. Then, at the receiver, a deep convolutional GAN decoder reconstructs the original CSI from this compressed vector with the discriminator assessing its quality. The evaluations show that the normalized MSE of the proposed method is $-7.05 \mathrm{~dB}$, which is lower than $-2.46 \mathrm{~dB}$ of CS-CsiNet based on deep learning, when the compression ratio is $1 / 64$. Besides GANs, VAEs are also suitable for this task. These CSI compression methods show excellent reconstruction accuracy across varying compression ratios, providing support to reduce the overhead of CSI transmission and storage.
2) Beamforming: Beamforming is a critical element in ISAC systems, significantly affecting the scanning accuracy in sensing tasks. Adaptive beam alignment remains a central challenge in this area. To address this, the authors in [11] introduce a VAE based dual timescale learning and adaptation framework. For the long timescale, a deep recurrent VAE (DRVAE) is proposed to learn a probabilistic model of beam dynamics based on noisy beam-training observations. For the short timescale, an adaptive beam-training procedure is formulated as a partially observable Markov decision process. This is then optimized using point-based value iteration by leveraging both beam-training feedbacks and probabilistic predictions of the strongest beam pair provided by the DRVAE. The proposed DR-VAE approach achieves near-optimal spectral efficiency, with a gain of $85 \%$ over a conventional strategy that scans exhaustively over the dominant beam pairs.
In ISAC, such a method not only minimizes the overhead associated with beam alignment during sensing process, but also boosts spectral efficiency, thereby increasing communication throughput.
3) Channel Estimation: Channel estimation is important for sensing reliability, particularly in sensing systems that rely on CSI. Diffusion models, excel at learning high-dimensional gradients and model the log distribution of the data, are wellsuited for modeling high-dimensional millimeter-wave MIMO channels. In [12], the authors introduce a MIMO channel estimation method using score-based diffusion models. They first train a score-based generative model in an unsupervised manner using a database of known channels, which is independent of pilot symbols. Then, annealed Langevin dynamics is used for channel estimation by sampling from the posterior distribution. Compared to conventional supervised deep learning methods, this approach can offer a communication gain of up to $5 \mathrm{~dB}$ to the end-to-end coded communication system can reach up to $5 \mathrm{~dB}$. More importantly, within ISAC systems, this approach holds the potential to solve the problems of estimating the channel in an out-of-distribution setting, i.e., the environments not seen during training, thereby providing more robust data support for the CSI-based sensing in complex channel conditions.
4) Signal Enhancement: Signal parameter estimation is crucial for wireless sensing in ISAC systems, as it provides valuable observations for tasks like target detection and localization. Estimating signal parameters in low SNR conditions is particularly challenging. One effective strategy to address this issue is to improve the SNR using the generative capabilities of GAI models. Hence, in [13], the authors convert lowSNR complex signals into images. Then, they employ a Unet structure as the GAN's generator to encode these images, effectively boosting the SNR. After that, PatchGAN, i.e., the discriminator, assesses the quality of the enhanced image. This approach successfully increases the SNR of the signal thereby yielding more accurate parameter estimation. Adapting this concept to ISAC, incomplete and low-SNR signals can be converted into images. GAI models, once trained, can then refine these images, effectively boosting the signal SNR and thereby improving parameter estimation and sensing performance.

Besides the aforementioned applications, GAI can also be applied to sensing signal processing. For instance, in [14], the Transformer is used to capture inter-feature correlations among received signal strength observations, thereby boosting the multi-target localization capability. We summarize the above observations in Table II

## C. Discussion

So far, various GAI models have been integrated into the physical layer, offering potential support for both the communication and sensing of ISAC systems from diverse perspectives. From the above investigations, we can see that the designs leverage the following prominent capabilities of GAI:

- Capability of capturing complex data distributions. For intricate datasets with complex distributions that are

TABLE II

THE USE OF GAI IN THE PHYSICAL LAYER AND ITS POTENTIAL SUPPORT FOR COMMUNICATIONS OF ISAC. BLUE CELLS REPRESENT THE DISCUSSED CONTENT, WHITE CELLS REFERENCE OTHER WORKS, AND EMPTY CELLS DENOTE UNEXPLORED AREAS.

| Issues | Model layer |  |  |  | ISAC application layer <br> Communication \& sensing perspectives |
| :---: | :---: | :---: | :---: | :---: | :---: |
|  | GANs | NFs | VAEs | DFMs |  |
| Multiple access | Source data enhancement | - | - | - | Potential benefits for communication: <br> a. Stronger signal detection capabilities in systems <br> with unknown channel noise <br> b. More secure communication with lower BER <br> c. Better anomaly signal detection capabilities <br> d. Stronger spoofing signal generation and defense <br> capabilities <br> e. More efficient coding with higher PSNR <br> f. Enhanced access capabilities for multiple devices |
| Signal detection | Learn the channel <br> transition probability | Model unanalyzable <br> system noise | Learn the probability <br> distribution of the <br> input signal | Signal power spectral <br> density generation |  |
| Communication <br> security | Spoofing signal generation | - | Handle the influence of <br> random codeword <br> variations | - |  |
| Coding | Codebook design | - | Source data dimension <br> transformation | Channel distribution <br> generation |  |
| CSI compression | CSI data compression and <br> decompression | - | - | - | Potential benefits for sensing: <br> a. Superior data compression ratio and improved <br> reconstruction accuracy <br> b. Advanced CSI estimation accuracy for sensing <br> c. Enhanced beamforming performance with lower <br> overhead for beam alignment <br> d. Repair and generate the sensing signal |
| Beamforming | Map the channels for <br> precoder extraction | - | Learn the distribution <br> of the dynamic beams | - |  |
| Channel estimation | Model the complex <br> channel distribution | - | Model unknown <br> channel distribution | Learn the distribution <br> of wireless channel |  |
| Signal enhancement | Synthetic micro-Doppler <br> spectrum signature | - | - | Produce and recover <br> the denoised channel |  |

difficult, or even impossible to analyze directly, such as the noise and dynamic features of users, GAI models can be employed to capture their latent distributions. On this basis, the acquired distributions can be sampled, thereby supporting corresponding physical layer technologies, like signal detection in the system with complex noise and beam prediction in dynamic environments.

- Capability of transforming and processing data across various dimensions. For high-dimensional data, GAI models can reduce its dimensionality through encoding and subsequently decode it to recover the original highdimensional data. This facilitates the efficient compression, storage, and transmission of high-dimensional data within the ISAC system. For data with simpler distributions, GAI models can project them to more complex target spaces, thereby aiding in more efficient sampling and more accurate density estimation.
- Capability of restoring and enhancing data. For data in the ISAC system with a low SNR, such as the covariance matrix of received signals with low SNR as mentioned earlier, GAI models can effectively restore them. This restoration contributes to enhanced outcomes in subsequent stages, like more precise parameter estimation. Moreover, the generative capabilities of GAI can also recover incomplete data, ensuring that the subsequent processing can be effectively carried out.


## IV. CASE STUDY

Signal DoA estimation, which helps in identifying the location of the signal source, is crucial in both near-field and farfield ISAC systems. Besides, it also facilitates beamforming, enhancing the active near-field communication (NFC) [15]. However, when the antenna spacing exceeds half the wavelength (i.e., $\lambda$ ), the DoA estimation becomes challenging due to phase ambiguity. In this section, we show how to use GAI, i.e., diffusion models, to address this challenge, thereby providing support for near-field ISAC.

## A. Problem Description

Based on a uniform linear array (ULA), the DoA estimation relies on the phase difference across signals received by adjacent antennas in the array. This phase difference is a function of both the distance and direction of the signal source relative to the ULA. In the near-field context, the phase difference has a one-to-one correspondence with the DoA when the antenna spacing is less than or equal to $0.25 \lambda$, allowing for effective DoA estimation, as shown by the clear signal spectrum in Fig. 2. However, as the antenna spacing expands, for instance, to $\lambda$, the increased propagation path length causes the phase ambiguity, leading to an ambiguous spectrum, as shown in Fig. 2 Under these conditions, the system is unable to identify the true signal DoA, which affects subsequent tasks like localization and beamforming.

## B. Proposed Design

Essentially, the signal spectrum is a matrix and the distribution of data in it describes the signal DoA in the near-field. When the antenna spacing in the array exceeds half the wavelength, the signal spectrum becomes ambiguous, indicating a change in the data distribution and leading to an inability to recognize the correct signal DoA. The diffusion model's powerful inference capabilities can be employed to explore the relationship between ambiguous and clear signal spectra. Thereby, we propose a diffusion model-based signal spectrum generator (SSG), the architecture of which is illustrated in Fig. 3

In the near-field scenario (shown in Fig. 2) with $N=4$ and $d=0.5 \lambda$, we produce 10,000 paired signal spectra via simulation, assigning $80 \%$ for training and $20 \%$ for testing. During the simulation, the number of signal sources is fixed at 3, with the corresponding DoAs and ranges randomly generated within $0-180$ degrees and $0-6 \lambda$, respectively, and the SNR is also randomly generated between $0-5 \mathrm{~dB}$. To ensure data consistency, the ambiguous spectra are obtained

![](https://cdn.mathpix.com/cropped/2024_06_04_c7e6d93cb290df792351g-7.jpg?height=1347&width=1800&top_left_y=186&top_left_x=152)

Fig. 2. The cause of ambiguous signal spectrum and its impact on applications. Here, $\lambda$ is the signal wavelength, $d$ is the antenna spacing, $\theta$ is the DoA, $r$ is the distance between the signal source and the reference antenna, $2 N+1$ is the total number of antennas. When $d$ is less than half of the $\lambda$, the signal DoA can be accurately estimated. However, as $d$ increases, for instance, to $\lambda$ or $1.5 \lambda$, the signal spectrum becomes ambiguous, obstructing the identification of the true signal DoA and subsequently impacting further operations such as localization and beamforming."

via DoA estimation using the signals captured by antennas with odd index (corresponding to an antenna spacing of $\lambda$ ), while signals from antennas 3 to 7 generate the corresponding correct spectra. Subsequently, the ambiguous spectrum serves as the observation, while the correct spectrum acts as the expert solution for training the SSG. During the training, the SSG adds noise to the expert solution and subsequently denoises it step by step, as shown in Steps 4 and 5, refining the denoising network hyperparameters along the way. These hyperparameters establish the denoising criteria, guiding the diffusion model's inference based on the observations. Therefore, after training, the SSG can generate the expert solutions (clear signal spectra) based on the given observations (ambiguous signal spectra) via the trained denoising network.

## C. Performance Evaluation

The Part (i)-a in Fig. 4 presents the test reward curve. According to the results, the difference between the solution generated by the SSG and the real expert solution gradually narrows over training, indicating that the SSG can learn the denoising network's hyperparameters through the noising and denoising processes and can subsequently utilize the denoising network to generate the corresponding expert solution. Furthermore, the test reward of SSG stabilizes around -10, better than -80 of DRL [2] based method, indicating SSG is better than DRL in signal spectrum reconstruction. This may be due to the DRL struggling to focus on the key points in the spectrum corresponding to the DoAs, thereby failing to effectively learn the correct solution.

The results in Part (ii) of Fig. 4 illustrate the expert solution generation process. As can be seen, the trained SSG can effectively produce the expert solution through sequential denoising based on the ambiguous spectrum in Fig. 4 Part (i)-b. The generated signal spectrum in Part (i)-d depicts the DoAs of the three signal sources are 31,99 , and 146 degrees, which closely align with the expert solution, shown in Part (i)-c, of 30, 99, and 146 degrees, respectively. Based on the generated signal spectrum and the corresponding ground truth, we observe that the SSG achieves a DoA estimation MSE of about 1.03

![](https://cdn.mathpix.com/cropped/2024_06_04_c7e6d93cb290df792351g-8.jpg?height=911&width=1781&top_left_y=195&top_left_x=172)

Fig. 3. The structure of the proposed SSG. In Step 1, the current observation, i.e., the ambiguous signal spectrum, is obtained. In Step 2, the corresponding expert solution is obtained. Steps 3-6 detail the training process via forward and backward diffusion. Using the expert solution, the loss function is designed to minimize the discrepancy between the generated signal spectrum and the expert solution.

degrees. This further proves that SSG can effectively produce the clear signal spectrum, which can be leveraged to both improve the energy efficiency of beamforming and reduce communication power consumption.

In addition, we investigate the impact of SSD on localization performance. During the test, we assume that the range is accurately estimated, and the system uses the DoA of the three peak points with the largest amplitude and the ranges to achieve localization. In Part (i)-e, results indicate a median localization error of about $1.25 \lambda$ without using SSG. However, the use of SSG reduces this error to around $0.21 \lambda$. This is intuitive, as ambiguous spectrums can lead the system to conduct localization using the incorrect DoA, causing notable errors.

## V. FUTURE DiRECTIONS

## A. GAI Application Security

While GAI has demonstrated its potential in the physical layer, it also poses certain risks. For instance, attacks on the training datasets can lead to training non-convergence or even failure, thereby wasting significant computational resources. Attacks on GAI model itself could cause more severe consequences, such as ineffective channel estimation and coding, ultimately impacting the ISAC performance. Hence, future research should address these security issues from both the dataset and model perspectives. Blockchain technology can ensure data authenticity and provider reliability, while offering a unified management for multi-party data, hence serving as one effective approach to resolving these security issues.

## B. Resource Allocation

The training and operation of GAI models consume computational, storage, and communication resources, disrupting the resource balance of the original system. Hence, integrating GAI models into the physical layer necessitates reallocating resources to ensure stable system operation. When local resources are abundant, strategies should be developed to maximize benefits while minimizing resource consumption based on task complexity and real-time requirements. When local resources are constrained, incentivization mechanisms, such as dynamic spectrum access, should be considered to ensure functional effectiveness, and then maximize benefits.

## C. Cell Free ISAC

The decentralized architecture of cell-free massive MIMO effectively reduces the distance between the access point and the user, thereby minimizing path loss. This configuration is naturally conducive to the utilization of millimeter wave and terahertz frequencies for ISAC performance. Within this framework, GAI can be utilized to optimize factors such as precoding and combining. This integration has the potential to generate high-gain, narrow beams in a mobile cell-free setting, further enhancing the efficacy of both target tracking and highcapacity wireless fronthaul.

## VI. CONCLUSION

In this article, we investigated GAI's use in the physical layer from various perspectives. We concluded that these applications primarily leverage GAI's capabilities in complex data feature extraction, transformation, and enhancement.

![](https://cdn.mathpix.com/cropped/2024_06_04_c7e6d93cb290df792351g-9.jpg?height=933&width=1791&top_left_y=184&top_left_x=167)

Fig. 4. The experimental results. The Part (i) describes the training process of SSG as well as the comparison among the final generated signal spectrum, the observed ambiguous spectrum, and the real signal spectrum. The results presented in Part (ii) detail the signal spectrum generation process of the proposed SSG. During the inference process, the SSG starts with noise an uses the trained denoising network to denoise it. Therefore, as the number of inference steps increases, the noise in the spectrum gradually diminishes. Finally, after 10 inference steps, the clear signal spectrum is obtained.

Subsequently, we analyzed how GAI-enhanced physical layer technologies can potentially support ISAC systems, considering both sensing and communication aspects. In the case study, we introduced the diffusion model based SSG. Operating in the physical layer, SSG addresses the DoA estimation problem that arises when array spacing exceeds half the wavelength. These insights emphasize the crucial role of GAI in the ISAC physical layer and the pressing need for a further exploration of its applications.

## REFERENCES

[1] S. Bond-Taylor, A. Leach, Y. Long, and C. G. Willcocks, "Deep generative modelling: A comparative review of vaes, gans, normalizing flows, energy-based and autoregressive models," IEEE transactions on pattern analysis and machine intelligence, vol. 44, pp. 732-7347, 2021.

[2] H. Du, R. Zhang, Y. Liu, J. Wang, Y. Lin, Z. Li, D. Niyato, J. Kang, Z. Xiong, S. Cui et al., "Beyond deep reinforcement learning: A tutorial on generative diffusion models in network optimization," arXiv preprint arXiv:2308.05384, 2023.

[3] L. Zhang, A. Rao, and M. Agrawala, "Adding conditional control to text-to-image diffusion models," IEEE International Conference on Computer Vision (ICCV), 2023.

[4] L. Sun, Y. Wang, A. L. Swindlehurst, and X. Tang, "Generativeadversarial-network enabled signal detection for communication systems with unknown channel models," IEEE Journal on Selected Areas in Communications, vol. 39, no. 1, pp. 47-60, 2020.

[5] Y. Cui, F. Liu, X. Jing, and J. Mu, "Integrating sensing and communications for ubiquitous IoT: Applications, trends, and challenges," IEEE Network, vol. 35, no. 5, pp. 158-167, 2021.

[6] K. He, L. He, L. Fan, Y. Deng, G. K. Karagiannidis, and A. Nallanathan, "Learning-based signal detection for MIMO systems with unknown noise statistics," IEEE Transactions on Communications, vol. 69, no. 5, pp. 3025-3038, 2021.
[7] C.-H. Lin, C.-C. Wu, K.-F. Chen, and T.-S. Lee, "A variational autoencoder-based secure transceiver design using deep learning," in GLOBECOM 2020-2020 IEEE Global Communications Conference. IEEE, 2020, pp. 1-7.

[8] C. Duan, S. Zhang, P. Yin, X. Li, and J. Luo, "SCMA-TPGAN: A new perspective on sparse codebook multiple access for UAV system," Computer Communications, vol. 200, pp. 161-170, 2023.

[9] Y. M. Saidutta, A. Abdi, and F. Fekri, "Joint source-channel coding over additive noise analog channels using mixture of variational autoencoders," IEEE Journal on Selected Areas in Communications, vol. 39, no. 7, pp. 2000-2013, 2021.

[10] B. Tolba, M. Elsabrouty, M. G. Abdu-Aguye, H. Gacanin, and H. M. Kasem, "Massive MIMO CSI feedback based on generative adversarial network," IEEE Communications Letters, vol. 24, no. 12, pp. 2805-2808, 2020 .

[11] M. Hussain and N. Michelusi, "Adaptive beam alignment in mm-wave networks: A deep variational autoencoder architecture," in 2021 IEEE Global Communications Conference (GLOBECOM). IEEE, 2021, pp. $1-6$.

[12] M. Arvinte and J. I. Tamir, "MIMO channel estimation using score-based generative models," IEEE Transactions on Wireless Communications, 2022.

[13] X. Cao, F. Wang, B. Yi, Z. Wei, and L. Liu, "Pix2pix-based doa estimation with low snr," in 2022 IEEE 10th Asia-Pacific Conference on Antennas and Propagation (APCAP). IEEE, 2022, pp. 1-2.

[14] Z. Lu, H. Liu, and X. Zhang, "Radio tomographic imaging localization based on transformer model," in 2023 IEEE 6th Information Technology, Networking, Electronic and Automation Control Conference (ITNEC), vol. 6. IEEE, 2023, pp. 1134-1138.

[15] Y. Liu, Z. Wang, J. Xu, C. Ouyang, X. Mu, and R. Schober, "Near-field communications: A tutorial review," arXiv preprint arXiv:2305.17751, 2023.

</end of paper 0>


<paper 1>
# Unleashing the Power of Edge-Cloud Generative AI in Mobile Networks: A Survey of AIGC Services 

Minrui Xu, Hongyang Du*, Dusit Niyato, Fellow, IEEE, Jiawen Kang, Zehui Xiong, Shiwen Mao, Fellow, IEEE,<br>Zhu Han, Fellow, IEEE, Abbas Jamalipour, Fellow, IEEE, Dong In Kim, Fellow, IEEE, Xuemin (Sherman) Shen,<br>Fellow, IEEE, Victor C. M. Leung, Life Fellow, IEEE, and H. Vincent Poor, Life Fellow, IEEE


#### Abstract

Artificial Intelligence-Generated Content (AIGC) is an automated method for generating, manipulating, and modifying valuable and diverse data using AI algorithms creatively. This survey paper focuses on the deployment of AIGC applications, e.g., ChatGPT and Dall-E, at mobile edge networks, namely mobile AIGC networks, that provide personalized and customized AIGC services in real time while maintaining user privacy. We begin by introducing the background and fundamentals of generative models and the lifecycle of AIGC services at mobile AIGC networks, which includes data collection, training, finetuning, inference, and product management. We then discuss the collaborative cloud-edge-mobile infrastructure and technologies required to support AIGC services and enable users to access AIGC at mobile edge networks. Furthermore, we explore AIGCdriven creative applications and use cases for mobile AIGC networks. Additionally, we discuss the implementation, security, and privacy challenges of deploying mobile AIGC networks. Finally, we highlight some future research directions and open issues for the full realization of mobile AIGC networks.


Index Terms-AIGC, Generative AI, Mobile edge networks, Communication and Networking, AI training and inference, Internet technology

M. Xu, H. Du, and D. Niyato are with the School of Computer Science and Engineering, Nanyang Technological University, Singapore 639798, Singapore (e-mail: minrui001@e.ntu.edu.sg; hongyang001@e.ntu.edu.sg; dniyato@ntu.edu.sg).

J. Kang is with the School of Automation, Guangdong University of Technology, and Key Laboratory of Intelligent Information Processing and System Integration of IoT, Ministry of Education, Guangzhou 510006, China, and also with Guangdong-HongKong-Macao Joint Laboratory for Smart Discrete Manufacturing, Guangzhou 510006, China (e-mail: kavinkang@gdut.edu.cn).

Z. Xiong is with the Pillar of Information Systems Technology and Design, Singapore University of Technology and Design, Singapore 487372, Singapore (e-mail: zehui_xiong @ sutd.edu.sg).

S. Mao is with the Department of Electrical and Computer Engineering, Auburn University, Auburn, AL 36849-5201 USA (email: smao@ ieee.org).

Z. Han is with the Department of Electrical and Computer Engineering, University of Houston, Houston, TX 77004 USA, and also with the Department of Computer Science and Engineering, Kyung Hee University, Seoul 446-701, South Korea (e-mail: zhan2 @uh.edu).

A. Jamalipour is with the School of Electrical and Information Engineering, University of Sydney, Sydney, NSW 2006, Australia (e-mail: a.jamalipour@ ieee.org).

D. I. Kim is with the Department of Electrical and Computer Engineering, Sungkyunkwan University, Suwon 16419, South Korea (email:dikim@skku.ac.kr).

X. Shen is with the Department of Electrical and Computer Engineering, University of Waterloo, Waterloo, ON N2L 3G1, Canada (e-mail: sshen@uwaterloo.ca).

V. C. M. Leung is with the College of Computer Science and Software Engineering, Shenzhen University, Shenzhen 518061, China, and also with the Department of Electrical and Computer Engineering, The University of British Columbia, Vancouver BC V6T 1Z4, Canada (E-mail: vleung @ ieee.org).

H. V. Poor is with the Department of Electrical and Computer Engineering, Princeton University, Princeton, NJ 08544, USA (e-mail: poor @ princeton.edu).

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-01.jpg?height=889&width=876&top_left_y=688&top_left_x=1080)

Fig. 1: The overview of mobile AIGC networks, including the cloud layer, the edge layer, and the mobile device layer. The lifecycle of AIGC services, including data collection, pretraining, fine-tuning, inference, and product management, is circulated among the core networks and edge networks.

## I. INTRODUCTION

## A. Background

In recent years, artificial intelligence-generated content (AIGC) has emerged as a novel approach to the production, manipulation, and modification of data [1]. By utilizing AI technologies, AIGC automates content generation alongside traditionally professionally-generated content (PGC) and usergenerated content (UGC) [2]-[4]. With the marginal cost of data creation reduced to nearly zero, AIGC, e.g., ChatGPT [5], promises to supply a vast amount of synthetic data for AI development and the digital economy, offering significant productivity and economic value to society. The rapid growth of AIGC capabilities is driven by the continuous advancements in AI technology, particularly in the areas of large-scale and multimodal models [6], [7]. A prime example of this progress is the development of the transformer-based DALL-E [8]
which is designed to generate images by predicting successive pixels. In its latest iteration, DALL-E2 [9], a diffusion model is employed to reduce noise generated during the training process, leading to more refined and novel image generation. In the context of text-to-image generation using generative AI models, the language model serves as a guide, enhancing semantic coherence between the input prompt and the resulting image. Simultaneously, the generative AI model processes existing image attributes and components, generating limitless synthesis images from existing datasets.

Based on large-scale pre-trained models with billions of parameters, AIGC services are designed to enhance knowledge and creative work fields that employ billions of people. By leveraging generative $\mathrm{AI}$, these fields can achieve at least a $10 \%$ increase in efficiency for content creation, potentially generating trillions of dollars in economic value [10]. AIGC can be applied to various forms of text generation, ranging from practical applications, such as customer service inquiries and messages, to creative tasks like activity tracking and marketing copywriting [11]. For example, OpenAI's ChatGPT [12] can automate the generation of socially valuable content based on user-provided prompts. Through extended and coherent conversations with ChatGPT, individuals from diverse professions from all walks of life, can seek assistance in debugging code, discovering healthy recipes, writing scripts, and devising marketing campaigns. In the realm of image generation, generative AI models can process existing images according to their attributes and components, enabling end-toend image synthesis, such as generating complete images directly from existing ones [9]. Moreover, generative AI models hold immense potential for cross-modal generation, as they can spatially process existing video attributes and simultaneously process multiple video clips automatically [13].

The benefits of AIGC in content creation, when compared to PGC and UGC, are already apparent to the public. Specifically, generative AI models can produce high-quality content within seconds and deliver personalized content tailored to users' needs [3], [14]. Over time, the performance of AIGC has significantly improved, driven by enhanced models, increased data availability, and greater computational power [15]. On one hand, superior models [6], such as diffusion models, have been developed to provide more robust tools for crossmodal AIGC generation. These advancements are attributed to the foundational research in generative AI models and the continuous refinement of learning paradigms and network structures within generative deep neural networks (DNNs). On the other hand, data and computing power for generative AI training and inference have become more accessible as networks grow increasingly interconnected [11], [16], [17]. For instance, generative AI models that require thousands of GPUs can be trained and executed in cloud data centers, enabling users to submit frequent data generation requests over core networks.

## B. Motivation

Although AIGC is acknowledged for its potential to revolutionize existing production processes, users accessing AIGC services on mobile devices currently lack support for interactive and resource-intensive data generation services [1], [18], [29]. Initially, the robust computing capabilities of cloud data centers can be utilized to pre-train generative $\mathrm{AI}$ models, such as GPT-3 for ChatGPT and GPT-4 for ChatGPT Plus. Subsequently, users can access cloud-based AIGC services via the core network by executing generative $\mathrm{AI}$ models on cloud servers. However, due to their remote nature, cloud services exhibit high latency. Consequently, deploying interaction-intensive AIGC services on mobile edge networks, i.e., mobile AIGC networks, as shown in Fig. 1. should be considered a more practical option [30]-[32]. In mobile AIGC networks, the cloud layer handles the pre-training and finetuning of AIGC models, which require a significant amount of computing and storage resources. In addition, the edge layer is responsible for data collection, inference, and product management, requiring specialized hardware and software, as well as efficient communication protocols. Finally, the mobile device layer is crucial for data collection, inference, and product management with low latency, presenting unique challenges that can be addressed with specialized techniques such as federated learning and differential privacy. In detail, the motivations for developing mobile AIGC networks include

- Low-latency: Instead of directing requests for AIGC services to cloud servers within the core network, users can access low-latency services in mobile AIGC networks [33]. For example, users can obtain AIGC services directly in radio access networks (RANs) by downloading pre-trained models to edge servers and mobile devices for fine-tuning and inference, thereby supporting real-time, interactive AIGC.
- Localization and Mobility: In mobile AIGC networks, base stations with computing servers at the network's edge can fine-tune pre-trained models by localizing service requests [34], [35]. Furthermore, users' locations can serve as input for AIGC fine-tuning and inference, addressing specific geographical demands. Additionally, user mobility can be integrated into the AIGC service provisioning process, enabling dynamic and reliable AIGC service provisioning.
- Customization and Personalization: Local edge servers can adapt to local user requirements and allow users to request personalized services based on their preferences while providing customized services according to local service environments. On one hand, edge servers can tailor AIGC services to the needs of the local user community by fine-tuning them accordingly [3]. On the other hand, users can request personalized services from edge servers by specifying their preferences.
- Privacy and Security: AIGC users only need to submit service requests to edge servers, rather than sending preferences to cloud servers within the core network. Therefore, the privacy and security of AIGC users can be preserved during the provisioning, including fine-tuning and inference, of AIGC services.

As illustrated in Fig. 1, when users access AIGC services on mobile edge networks through edge servers and mobile

TABLE I: Summary of related works versus our survey.

| Year | Ref. | Contributions | AIGC <br> Algorithms | AIGC <br> Applications | Edge <br> Intelligence |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 2019 | \|18| | Introduce mobile edge intelligence, and discuss the infrastructure, <br> implementation methodologies, and use cases | $x$ | $x$ | $\checkmark$ |
| 2020 | 19 | Present the implementation challenges of FL at mobile edge <br> networks | $\checkmark$ | $x$ | $\checkmark$ |
|  | 15 | Discuss the visions, implementation details, and applications of <br> the convergence of edge computing and DL | $\checkmark$ | $x$ | $\checkmark$ |
| 2021 | $\overline{20}$ | Investigate the copyright laws regarding AI-generated music | $\checkmark$ | $\bar{\checkmark}$ | $x$ |
|  | $\sqrt{2]}$ | Illustrate the interaction of art and AI from two perspectives, i.e., <br> AI for art analysis and AI for art creation | $x$ | $\checkmark$ | $x$ |
|  | [3] | Discuss the application of computational arts in Metaverse to <br> create surrealistic cyberspace | $\checkmark$ | $\checkmark$ | $x$ |
|  | 21 | Investigate the deployment of distributed learning in wireless <br> networks | $x$ | $x$ | $\checkmark$ |
|  | $\mid 22$ | Provide a comprehensive overview of the major approaches, <br> datasets, and metrics used to synthesize and process multimodal <br> images | $\checkmark$ | $\checkmark$ | $x$ |
|  | $\mid 23$ | Propose a novel conceptual architecture for 6G networks, which <br> consists of holistic network virtualization and pervasive network <br> intelligence | $x$ | $x$ | $\checkmark$ |
| 2022 | $\mid 24$ | Discusses the visions and potentials of low-power, low-latency, <br> reliable, and trustworthy edge intelligence for $6 \mathrm{G}$ wireless <br> networks | $x$ | $x$ | $\checkmark$ |
|  | [6] | Provide comprehensive guidance and comparison among <br> advanced generative models, including GAN, energy-based <br> models, VAE, autoregressive models, flow-based models, and <br> diffusion models | $\checkmark$ | $x$ | $x$ |
|  | 25 | Present fundamental algorithms, classification and applications of <br> diffusion models | $\checkmark$ | $x$ | $x$ |
|  | 11 | Provide a comprehensive overview of generation and detection <br> methods for machine-generated text | $\checkmark$ | $\checkmark$ | $x$ |
|  | $\mid 26$ | Provide a comprehensive examination of what, why, and how <br> edge intelligence and blockchain can be integrated | $x$ | $x$ | $\checkmark$ |
|  | 27 | Introduce the architecture of edge-enabled Metaverse and discuss <br> enabling technologies in communication, computing, and <br> blockchain | $x$ | $\checkmark$ | $\checkmark$ |
| 2023 | 28 | Summarize existing works on the generation of gestures with <br> simultaneous speeches based on deep generative models | $\checkmark$ | $\checkmark$ | $x$ |
|  | $[1]$ | A comprehensive tutorial on applying generative diffusion model <br> in various network optimization tasks. Case studies explore <br> integrating the diffusion model with DRL, incentive mechanism <br> design, semantic communications, and Internet of Vehicles (IoV) <br> networks. | $\checkmark$ | $x$ | $\checkmark$ |
|  | Ours | Investigate the deployment of mobile AIGC networks via <br> collaborative cloud-edge-mobile infrastructure, discuss creative <br> mobile applications and exemplary use cases, and identify <br> existing implementation challenges | $\checkmark$ | $\checkmark$ | $\checkmark$ |

devices, limited computing, communication, and storage resources pose challenges for delivering interactive and resourceintensive AIGC services. First, resource allocation on edge servers must balance the tradeoff among accuracy, latency, and energy consumption of AIGC services at edge servers. In addition, computationally intensive AIGC tasks can be offloaded from mobile devices to edge servers, improving inference latency and service reliability. Moreover, AI models that generate content can be cached in edge networks, similar to content delivery networks (CDNs) [36], [37], to minimize delays in accessing the model. Finally, mobility management and incentive mechanisms should be explored to encourage user participation in both space and time [38]. Compared to traditional AI, AIGC technology requires overall technical maturity, transparency, robustness, impartiality, and insightfulness of the algorithm for effective application implementation. From a sustainability perspective, AIGC can use both existing and synthetic datasets as raw materials for generating new data. However, when biased data are used as raw data, these biases persist in the knowledge of the model, which inevitably leads to unfair algorithm results. Finally, static generative AI models rely primarily on templates to generate machinegenerated content that may have similar text and output structures.

## C. Related Works and Contributions

In this survey, we provide an overview of research activities related to AIGC and mobile edge intelligence, as illustrated in

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-04.jpg?height=1640&width=1721&top_left_y=183&top_left_x=191)

Fig. 2: The development roadmap of AIGC and mobile edge networks from 2013 to Oct 2023. From the perspective of AIGC technology development, AIGC has evolved from generating text and audio to generating 3D content. From the perspective of mobile edge computing, computing has gradually shifted from cloud data centers to mobile device computing.

Fig. 2. Given the increasing interest in AIGC, several surveys on related topics have recently been published. Table $\square$ presents a comparison of these surveys with this paper.

The study by [1] offers a focused exploration of Generative Diffusion Models (GDMs) in network optimization tasks ${ }^{1}$. Commencing with an essential background on GDMs, it outlines their ability to model complex data distributions effectively. This enables them to excel in diverse tasks, ranging from image generation to reinforcement learning. The paper advances by presenting case studies that integrate GDMs with Deep Reinforcement Learning, incentive mechanism design,[^0]

Semantic Communications, and Internet of Vehicles networks. These case studies substantiate the model's practical utility in solving complex network optimization problems. The study in [39] provides a comprehensive overview of the current generative AI models published by researchers and the industry. The authors identify nine categories summarizing the evolution of generative AI models, including text-to-text, text-to-image, text-to-audio, text-to-video, text-to-3D, text-to-code, text-toscience, image-to-text, and other models. In addition, they reveal that only six organizations with enormous computing power and highly skilled and experienced teams can deploy these state-of-the-art models, which is even fewer than the number of categories. Following the taxonomy of generative

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-05.jpg?height=1594&width=1791&top_left_y=187&top_left_x=167)

Fig. 3: The outline of this survey, where we introduce the provisioning of AIGC services at mobile edge networks and highlight some essential implementation challenges about mobile edge networks for provisioning AIGC services.

AI models developed in [39], other surveys discuss generative AI models in detail subsequently. The study in [11] examines existing methods for generating text and detecting models. The study in [22] provides a comprehensive overview of the major approaches, datasets, and evaluation metrics for multimodal image synthesis and processing. Based on techniques of speech and image synthesis, the study in [28] summarizes existing works on the generation of gestures with simultaneous speeches based on deep generative models. The study in [20] investigates the copyright laws regarding AIgenerated music, which includes the complicated interactions among AI tools, developers, users, and the public domain. The study in [6] provides comprehensive guidance and comparison among advanced generative models, including GANs, energybased models, variational autoencoder (VAE), autoregressive models, flow-based models, and diffusion models. As diffusion models draw tremendous attention in generating creative data, the study in [25] presents fundamental algorithms and comprehensive classification for diffusion models. Based on these algorithms, the authors [2] illustrate the interaction of art and $\mathrm{AI}$ from two perspectives, i.e., AI for art analysis and AI for art creation. In addition, the authors in [3] discuss the application of computational arts in the Metaverse to create surrealistic cyberspace.

In 6G [23], mobile edge intelligence based on edge computing systems, including edge caching, edge computing, and edge intelligence, for intelligent mobile networks, is introduced in [18], [40]. The study in [21] investigates the deployment of distributed learning in wireless networks. The study [19] provides a guide to federated learning (FL) and a
comprehensive overview of implementing FL at mobile edge networks. The authors offer a detailed analysis of the challenges of implementing FL, including communication costs, resource allocation, privacy, and security. In [15], various application scenarios and technologies for edge intelligence and intelligent edges are presented and discussed in detail. In addition, the study [24] discusses the visions and potentials of low-power, low-latency, reliable, and trustworthy edge intelligence for $6 \mathrm{G}$ wireless networks. The study [26] explores how blockchain technologies can be used to enable edge intelligence and how edge intelligence can support the deployment of blockchain at mobile edge networks. The authors provide a comprehensive review of blockchain-driven edge intelligence, edge intelligence-amicable blockchain, and their implementation at mobile edge networks.

Distinct from existing surveys and tutorials, our survey concentrates on the deployment of mobile AIGC networks for real-time and privacy-preserving AIGC service provisioning. We introduce the current development of AIGC and collaborative infrastructure in mobile edge networks. Subsequently, we present the technologies of deep generative models and the workflow of provisioning AIGC services within mobile AIGC networks. Additionally, we showcase creative applications and several exemplary use cases. Furthermore, we identify implementation challenges, ranging from resource allocation to security and privacy, for the deployment of mobile AIGC networks. The contributions of our survey are as follows.

- We initially offer a tutorial that establishes the definition, lifecycle, models, and metrics of AIGC services. Then, we propose the mobile AIGC networks, i.e., provisioning AIGC services at mobile edge networks with collaborative mobile-edge-cloud communication, computing, and storage infrastructure.
- We present several use cases in mobile AIGC networks, encompassing creative AIGC applications for text, images, video, and 3D content generation. We summarize the advantages of constructing mobile AIGC networks based on these use cases.
- We identify crucial implementation challenges in the path to realizing mobile AIGC networks. The implementation challenges of mobile AIGC networks stem not only from dynamic channel conditions but also from the presence of meaningless content, insecure content precepts, and privacy leaks in AIGC services.
- Lastly, we discuss future research directions and open issues from the perspectives of networking and computing, machine learning (ML), and practical implementation considerations, respectively.

As the outline illustrated in Fig. 3, the survey is organized as follows. Section II examines the background and fundamentals of AIGC. Section III presents the technologies and collaborative infrastructure of mobile AIGC networks. The applications and advantages of mobile AIGC networks are discussed in Section IV, and potential use cases are shown in Section V. Section $V I$ addresses the implementation challenges. Section VII explores future research directions. Section VIII provides the conclusions.

## II. BACKGROUND AND FUNDAMENTALS OF AIGC

In this section, the background and fundamentals of AIGC technology are presented. Specifically, we examine the definition of AIGC, its classification, and the technological lifecycle of AIGC in mobile networks. Finally, we introduce ChatGPT as a use case, which is the most famous and revolutionary application of AIGC.

## A. Definitions of PGC, UGC, and AIGC

In the next generation of the Internet, i.e. Web 3.0 and Metaverse [41]-[43], there are three primary forms of content [2], including PGC, UGC, and AIGC.

1) Professionally-generated Content: PGC refers to professional-generated digital content [44]. Here, the generators are individuals or organizations with professional skills, knowledge, and experience in a particular field, e.g., journalists, editors, and designers. As these experts who create PGC are typically efficient and use specialized tools, PGC has the advantages in terms of automation and multimodality. However, because PGC is purposeful, the diversity and creativity of PGC can be limited.
2) User-generated Content: UGC refers to digital material generated by users, rather than by experts or organizations [45]. The users include website visitors and social media users. UGC can be presented in any format, including text, photos, video, and audio. The barrier for users to create UGC is being lowered. For example, some website ${ }^{2}$ allow users to create images with a high degree of freedom on a pixel-bypixel basis. As a result, UGC is more creative and diverse, thanks to a wide user base. However, UGC is less automated and less multimodal than the PGC that is generated by experts.
3) AIGC: AIGC is generated by using generative AI models according to input from users. Because AI models can learn the features and patterns of input data from the human artistic mind, they can develop a wide range of content. The recent success of text-to-image applications based on the diffusion model [46] and the ChatGPT based on transformer [12] has led to AIGC gaining a lot of attention. We have defined the AIGC according to its characteristics as follows

- Automatic: AIGC is generated by AI models automatically. After the AI model has been trained, users only need to provide input, such as the task description, to efficiently obtain the generated content. The process, from input to output, does not require user involvement and is done automatically by the AI models.
- Creativity: AIGC refers to an idea or item that is innovative. For example, AIGC is believed to be leading to the development of a new profession, called Prompt Engineer [47], which aims to improve human interaction with AI. In this context, the prompt serves as the starting point for the AI model, and it significantly impacts the originality and quality of the generated content. A wellcrafted prompt that is specific results in more relevant and creative content than a vague or general prompt.

${ }^{2}$ Example of a website that allows users to create their own UGC: https://ugc-nft.io/Home

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-07.jpg?height=586&width=879&top_left_y=192&top_left_x=165)

Fig. 4: The four development stages of ChatGPT, including pre-training, fine-tuning, inference, and product management.

- Multimodal: The AI models to generate AIGC can handle multimodal input and output. For example, ChatGPT [12] allows conversational services that employ text as input and output, DALL-E 2 [48] can create original, realistic images from a text description, and AIGC services with voice and 3D models as input or output are progressing [49].
- Diverse: AIGC is diverse in service personalization and customization. On the one hand, users can adjust the input to the AI model to suit their preferences and needs, resulting in a personalized output. On the other hand, AI models are trained to provide diverse outputs. For example, consider the DALL-E 2 as an example, the model can generate images of individuals that more correctly represent the diversity of the global population, even with the same text input.
- Extendedly valuable: AIGC should be extendedly valuable to society, economics, and humanity [50]. For example, AI models can be trained to write medical reports and interpret medical images, enabling healthcare personnel to make accurate diagnoses.

AIGC provides various advantages over PGC and UGC, including better efficiency, originality, diversity, and flexibility. The reason is that AI models can produce vast amounts of material quickly and develop original content based on established patterns and principles. These advantages have led to the growing creative applications of the generative AI models, which are discussed in Section IV-A1.

## B. Serving ChatGPT at Mobile Edge Networks

ChatGPT, developed by OpenAI, excels at generating human-like text and engaging in conversations [12]. Based on the GPT-3 [51], this transformer-based neural network model can produce remarkably coherent and contextually appropriate text. Among its primary advantages, ChatGPT is capable of answering questions, providing explanations, and assisting with various tasks in a manner nearly indistinguishable from human responses. As illustrated in Fig. 4, the development of ChatGPT involves four main stages, including pre-training, fine-tuning, inference, and product management.

1) Pre-training: In the initial stage, known as pre-training, the foundation model of ChatGPT, GPT-3, is trained on a large corpus of text, which includes books, articles, and other information sources. This process enables the model to acquire knowledge of language patterns and structures, as well as the relationships between words and phrases. The base model, GPT-3, is an autoregressive language model with a Transformer architecture that has 175 billion parameters, making it one of the largest language models available. During pre-training, GPT-3 is fed with a large corpus of text from diverse sources, such as books, articles, and websites for self-supervised learning, where the model learns to predict the next word in a sentence given the context. To train the foundation model, the technique used is called maximum likelihood estimation, where the model aims to maximize the probability of predicting the next word correctly. Training GPT-3 demands significant computational resources and time, typically involving specialized hardware like graphics processing units (GPUs) or tensor processing units (TPUs). The exact resources and time required depend on factors such as model size, dataset size, and optimization techniques.
2) Fine-tuning: The fine-tuning stage of ChatGPT involves adapting the model to a specific task or domain, such as customer service or technical support, to enhance its accuracy and relevance for that task. To transform ChatGPT into a conversational AI, a supervised learning process is employed using a dataset containing dialogues between humans and AI models [52]. To optimize ChatGPT's parameters, a reward model for reinforcement learning is built by ranking multiple model responses by quality. Alternative completions are ranked by AI trainers, and the model uses these rankings to improve its performance through several iterations of Proximal Policy Optimization [53]. This technique allows ChatGPT to learn from its mistakes and improve its responses over time.
3) Inference: In the inference stage, ChatGPT generates text based on a given input or prompt, testing the model's ability to produce coherent and contextually appropriate responses relevant to the input. ChatGPT generates responses by leveraging the knowledge it acquired during pre-training and fine-tuning, analyzing the context of the input to generate relevant and coherent responses. In-context learning involves analyzing the entire context of the input [54], including the dialogue history and user profile, to generate responses that are personalized and tailored to the user's needs. ChatGPT employs chain-of-thought to generate responses that are coherent and logical, ensuring that the generated text is not only contextually appropriate but also follows a logical flow. The resources consumed during inference are typically much lower than those required for training, making real-time applications and services based on ChatGPT computationally feasible.
4) Product Management: The final product management phase involves deploying the model in a production environment and ensuring its smooth and efficient operation. In the context of mobile edge networks, the applications of AI-powered tools such as the new Bing [55] and Office 365 Copilot [56] could be particularly useful due to their
ability to provide personalized and contextually appropriate responses while conserving resources. The new Bing offers a new type of search experience with AI-powered features such as detailed replies to complex questions, summarized answers, and personalized responses to follow-up questions, while Office 365 Copilot, powered by GPT-4 from OpenAI, assists with generating documents, emails, presentations, and other tasks in Microsoft 365 apps and services. These tools can be integrated into mobile edge networks with specialized techniques that balance performance and accuracy while preserving data integrity.

- New bing: The new Bing offers a set of AI-powered features that provide a new type of search experience, including detailed replies to complex questions, summarized answers, and personalized responses to followup questions. Bing also offers creative tools such as assistance with writing poems and stories. In the context of mobile edge networks, Bing's ability to consolidate reliable sources across the web and provide a single, summarized answer could be particularly useful for users with limited resources. Additionally, Bing's ability to generate personalized responses based on user behavior and preferences could improve the experience of users in mobile edge networks.
- Office 365 copilot: Microsoft has recently launched an AI-powered assistant named Office 365 Copilot, which can be summoned from the sidebar of Microsoft 365 apps and services. Copilot can help users generate documents, emails, and presentations, as well as provide assistance with features such as PivotTables in Excel. It can also transcribe meetings, remind users of missed items, and provide summaries of action items. However, when deploying Copilot in mobile edge networks, it is important to keep in mind the limited resources of these devices and to develop specialized techniques that can balance performance and accuracy while preserving data integrity.

In addition to the previously mentioned commercial applications, ChatGPT holds substantial commercial potential owing to its capacity for producing human-like text, which is characteristically coherent, pertinent, and contextually fitting. This language model can be fine-tuned to accommodate a diverse array of tasks and domains, rendering it highly adaptable for numerous applications. ChatGPT exhibits remarkable proficiency in comprehending and generating text across multiple languages. Consequently, it can facilitate various undertakings, such as composing emails, developing code, generating content, and offering explanations, ultimately leading to enhanced productivity. By automating an assortment of tasks and augmenting human capabilities, ChatGPT contributes to a paradigm shift like human work, fostering new opportunities and revolutionizing industries. In addition to ChatGPT, more use cases developed by various generative AI models are discussed in Section $\mathrm{V}$

## C. Life-cycle of AIGC at Mobile Edge Networks

AIGC has gained tremendous attention as a technology superior to PGC and UGC. However, the lifecycle of the AIGC is also more elaborate. In the following, we discuss the AIGC lifecycle with mobile edge network enablement:

1) Data Collection: Data collection is an integral component of AIGC and plays a significant role in defining the quality and diversity of the material created by AI systems [57]. The data used to train AI models influences the patterns and relationships that the AI models learn and, consequently, the output. There are several data collection techniques for AIGC:
- Crowdsourcing: Crowdsourcing is the process of acquiring information from a large number of individuals, generally via the use of online platforms [58]. Crowdsourced data may be used to train ML models for text and image generation, among other applications. One common example is the use of Amazon Mechanical Turk 3 , where individuals are paid to perform tasks such as annotating text or images, which can then be used to train generative AI models.
- Data Market: Another way to obtain data is to buy it from a data provider. For example, Datatang ${ }^{4}$ is a firm that offers high-quality datasets and customized data services to assist businesses in enhancing the performance of their AI models. By giving access to varied, high-quality data, Datatang enables organizations to train AI models that are more accurate and effective, resulting in enhanced business performance and results.
- Internet-of-Things (IoT) data collection: In IoT, edge devices can help to collect the data, e.g., Global Positioning System (GPS) records and wireless sensing data [59]. For example, mobile phone sensors can track the device's movement and location or users [60]. The sensors can be used to collect data on the location, speed, and direction of movement of the device. These data are important for the implementation of personalized generative AI models. In addition to these traditional data collection methods, large-scale datasets are specifically designed for training generative AI models. For instance, the LAION-400M dataset [61], a large-scale, non-curated dataset consisting of 400 million English (image, text) pairs, is used in training models like CLIP.
- Passive data collection can be achieved with the help of edge networks [62]. In the smart city, sensors can be placed at strategic locations, such as on lamp posts, buildings, or other structures, to collect data on various aspects of the city environment. The data obtained by the sensors might be used to train AI models, which could subsequently be utilized to produce insights on air quality, traffic flow, and pedestrian density. Using data obtained from air quality sensors, an AI model can be trained to forecast air quality. The model can then be used to create a real-time map of the city's air quality. This real-time map could be used to guide policy choices about the management of air quality, leading to the development of generative AI models that are capable of generating decision solutions for managing air quality.[^1]

After the data has been collected, the data is then used to train the generative AI model.

2) Pre-training: The collected data is used to train the generative AI model. In mobile networks, training is typically done by central servers with powerful computing power. During the training process, the generative model automatically learns the patterns and features in the data and predicts the target outcome. We introduce several generative AI technologies in Section III-B, including Generative Adversarial Networks (GANs), VAE, Flow-based models, and diffusion models. These different training techniques have different strengths and weaknesses. The choice of technique depends on the specific requirements of the AIGC task, the available data, the desired output, and the computational resources available. After training is complete, cloud data centers can accept requests uploaded by network users to perform subsequent fine-tuning and inference tasks. Alternatively, cloud data centers can deliver the trained generative AI models down to network edge servers, which can process user requests locally. It is important to note the substantial computational resources required for the pre-training of generative AI models. For instance, the pre-training process of the Stable Diffusion model, a largescale AI model developed by Stability AI, was conducted on a cloud cluster with 256 Nvidia A100 GPUs for about 150,000 hours, which equates to a cost of approximately $\$ 600,000$ (https://huggingface.co/CompVis/stable-diffusion-v1-4). This highlights the intensive computational demands of training such models.
3) Fine-tuning: Fine-tuning in AIGC is the process of adjusting a pre-trained generative AI model to new tasks or domains by including a modest quantity of extra data. This approach can be used to enhance the model's performance on a given task or in a specific area by adjusting the AI model's parameters to suit the new data better. In mobile networks, tasks of fine-tuning can be performed by the edge network, using the small-size dataset uploaded by mobile users.
4) Inference: Using the trained generative AI model, inference can be done, which involves generating the desired content based on the input. generative AI models are traditionally managed via centralized servers, such as the Hugging Face platform 63]. In this setting, a large number of users make requests to the central server, wait in line, and obtain the requested services. Researchers aim to install AIGC services on edge networks to prevent request congestion and optimize service latency. Edge devices have sufficient computational capacity for AIGC inference and are closer to consumers than central servers. Therefore, users can interact with devices with a reduced transmission delay. In addition, as AIGC services are dispersed to several edge devices, the latency can be significantly reduced.
5) Product Management: The preceding stages cover content generation. However, as an irreplaceable online property comparable to NFT, AIGC possesses unique ownership, copyright, and worth for each content. Consequently, the preservation and management of AIGC products should be incorporated into the AIGC life cycle. Specifically, we refer to the party requesting the production of the AIGC as producers, e.g., mobile users or companies, who hire AIGC generators, e.g., network servers, to perform the AIGC tasks. Then, the main process in AIGC product management includes:

- Distribution: After the content is generated in network edge servers, the producers acquire ownership of the AIGC products. Consequently, they have the right to distribute these products to social media or AIGC platforms through edge networks
- Trading: Since AIGC products are regarded as a novel kind of non-fungible digital properties, they can be traded. The trading process can be modeled as a fund ownership exchange between two parties.

To implement the aforementioned AIGC lifecycle in mobile networks, we further investigate the technical implementation of AIGC in the following section.

## III. TECHNOLOGIES and COLLABORatiVE InfRASTRUCTURE OF MOBILE AIGC NETWORKS

In this section, we delve into the technologies and collaborative infrastructure of mobile AIGC networks. This section aims to provide a comprehensive understanding of the rationale and objectives of edge computing systems designed to support AIGC. Before we explore the design of these systems, it is crucial to establish the performance metrics that measure whether the system can maximize user satisfaction and utility.

## A. Evaluation Metrics of Generative AI Models and Services

We first discuss several metrics for assessing the quality of generative AI models, which can be used by AIGC service providers and users in mobile networks.

1) Inception Score: The Inception Score (IS) can be used to measure the accuracy of images generated by generative AI models in the mobile network [64]. The IS is based on the premise that high-fidelity generated images should have high-class probabilities, which suggest a reliable classification model, and a low Kullback-Leibler (KL) divergence between the projected class probability and a reference class distribution. To compute the IS, an exponential function is applied to the KL divergence between the anticipated class probabilities and the reference class distribution. The resulting value is then averaged over all created photos to obtain the IS. A higher IS indicates better overall image quality.
2) Frechet Inception Distance: The Frechet Inception Distance (FID) has emerged as a well-established metric for evaluating the effectiveness of generative models, particularly GANs, in terms of image quality and diversity [65]. FID leverages a pre-trained Inception network to calculate the distance between actual and synthetic image embeddings. This metric can be used by generative AI model providers to evaluate the quality of their generative models in mobile networks. Additionally, users can assess the capabilities of AIGC service providers through multiple requests for services based on FID measurements. However, when evaluating conditional text-toimage synthesis, FID only measures the visual quality of the output images, ignoring the adequacy of their conditioning on the input text [66]. Thus, while FID is an excellent evaluation metric for assessing image quality and diversity, it is limited when applied to conditional text-to-image synthesis.
3) R-Precision: R-Precision is a standard metric to evaluate how AI-generated images align with text inputs [67]. In mobile networks, the generative AI model producers can retrieve matching text from 100 text candidates using the AI-generated image as a query. The R-Precision measures the proportion of relevant items retrieved among the top-R retrieved items, where $\mathrm{R}$ is typically set to 1 . Specifically, the Deep Attentional Multimodal Similarity Model (DAMSM) is commonly used to compute the text-image retrieval similarity score [68]. DAMSM maps each subregion of an image and its corresponding word in the sentence to a joint embedding space, allowing for the measurement of fine-grained imagetext similarity for retrieval. However, it should be noted that text-to-image generative AI models can directly optimize the DAMSM module used to calculate R-Precision. This results in the metric being model-specific and less objective, limiting the evaluation of generative AI models in mobile networks.
4) CLIP-R-Precision: CLIP-R-Precision is an assessment metric to address the model-specific character of the RPrecision metric [69]. Instead of the conventional DAMSM, the suggested measure uses the latest multimodal CLIP model [7] to obtain R-Precision scores. Here, CLIP is trained on a massive corpus of web-based image-caption pairings and is capable, via a contrastive aim, of bringing together the two embeddings (visual and linguistic). Thus, the CLIP-RPrecision can provide a more objective evaluation of text-toimage generative AI model performance in mobile networks.
5) Quality of Experience: The Quality of Experience (QoE) metric plays a critical role in evaluating the performance of AIGC in mobile network applications [70]. QoE measures user satisfaction with the generated content, considering factors such as visual quality, relevancy, and utility. Gathering and analyzing user surveys, interaction, and behavioral data are standard methods used to determine QoE. In addition, the definition of QoE can vary depending on the objectives of the mobile network system designer and the user group being considered. With the aid of QoE, AIGC performance can be improved, and new models can be created to meet user expectations. It is essential to account for QoE when analyzing the performance of AIGC in mobile network applications to ensure that the generated content meets user expectations and provides a great user experience.

Based on the aforementioned evaluation metrics, diverse and valuable synthetic data can be generated from deep generative models. Therefore, in the next section, we introduce several generative AI models for mobile AIGC networks.

## B. Generative AI Models

Generative AI models aim to understand and replicate the true data distribution of input data through iterative training. This understanding allows the generation of novel data that closely aligns with the original distribution. As depicted in Fig. 5, this section delves into five fundamental generative models: Generative Adversarial Networks (GANs), energybased models, Variational Autoencoders (VAEs), flow-based models, and diffusion models.

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-10.jpg?height=583&width=875&top_left_y=188&top_left_x=1075)

Fig. 5: The model architecture of generative AI models, including generative adversarial networks, energy-based models, variational autoencoder, flow-based models, and diffusion models.

1) Generative Adversarial Networks: GANs are a fundamental framework for AIGC, comprising a generative model and a discriminative model [71]. The generative network aims to generate data that is as realistic and similar to the original data as possible to deceive the discriminative model based on the data in the original dataset. Conversely, the discriminant model's task is to differentiate between real and fake instances. During the GAN training process, the two networks continually enhance their performance by competing against each other until they reach a stable equilibrium. The advantages and disadvantages of GANs can be summarized as follows [71]:

- Advantages:
- GANs can generate new data closely resembling the original dataset, making them useful for tasks such as image synthesis and text-to-image translation.
- The adversarial training process leads to continuous improvement in the performance of both the generative and discriminative models.


## - Disadvantages:

- GANs can be difficult to train because the two networks in a GAN, i.e., the generator and the discriminator, constantly compete against others, making training unstable and slow.
- GANs primarily augment the existing dataset rather than creating entirely new content, limiting their ability to generate new content with other modalities.

2) Energy-based Generative Models: Energy-based generative models are a class of generative models that represent input data using energy values [72]. These models define an energy function and then minimize the input data's energy value through optimization and training. This approach is easily comprehensible, and the models exhibit excellent flexibility and generalization ability in providing AIGC services. EBMs capture dependencies by associating an unnormalized probability scalar (energy) to each configuration of the combination of observed and latent variables. Inference consists
of finding latent variables that minimize the energy given a set of observed variables. The model learns a function that associates low energies with the latent variables' correct values and higher energies with incorrect values.
3) Variational Autoencoder: The VAE [73] is a type of generative models that consist of two primary components: an encoder and a decoder network. The encoder transforms the input data into a set of parameters (mean and variance) in a latent space. These parameters are then used to sample from the latent space, generating latent variables. The decoder takes these latent variables as input and generates new data. VAEs differ from GANs in their training methods. While GANs are trained using a supervised learning approach, VAEs employ an unsupervised learning approach. This difference is reflected in how they generate data. VAEs generate data by sampling from the learned distribution, while GANs approximate the data distribution using the generator network.
4) Flow-based Generative Models: Flow-based generative models [74] facilitate the data generation process by employing probabilistic flow formulations. Additionally, these models compute gradients during generation using backpropagation algorithms, enhancing training and learning efficiency. Consequently, flow-based models in mobile edge networks present several benefits. One such advantage is computational efficiency. Flow-based models can directly compute the probability density function during generation, circumventing resource-intensive calculations. This promotes more efficient computation within mobile edge networks.
5) Generative Diffusion Models: Diffusion models are likelihood-based models trained with Maximum Likelihood Estimation (MLE) [25], as opposed to GANs trained with a minimax game between the generator and the discriminator. Therefore, the pattern collapses and thus the training instabilities can be avoided. Specifically, diffusion models are inspired by non-equilibrium thermodynamics theory [1]. They learn the inverse diffusion process to construct the desired data sample from noise by defining a Markov chain of diffusion steps that gradually add random noise to the data. In addition, diffusion can mathematically transform the computational space of the model from pixel space to a low-dimensional space called latent space. This reduces the computational cost and time required and improves the training efficiency of the model. Unlike VAE or flow-based models, diffusion models are learned using a fixed procedure, and the hidden variables have high dimensions that are the same as the original data. This versatility and computational efficiency make diffusion models highly effective across a broad range of applications, including computer vision, natural language processing, audio synthesis, 3D modeling, and network optimization [1].
6) Large Language Models: Large language models (LLM), which consist of billions of parameters, are trained on large-scale datasets [75], and thus demonstrate the ability to handle various downstream tasks. LLMs can understand input prompts and generate human-like text in response. These models have greatly influenced our interaction with technology and have helped pave the way for advancements in artificial general intelligence. For instance, Google's PaLM-E [76] is an embodied language model that can handle tasks involving reasoning, visuals, and language. It can process multimodal sentences and transfer knowledge across domains, enabling it to perform tasks such as robot planning and embodied question answering.

In wireless networks, deploying LLMs faces several important issues from the perspectives of wireless communications, computing, and storage [77]. In terms of wireless communications, efficient utilization of computing and energy resources is crucial due to the large sizes of LLMs and the need to process vast amounts of data [78]. Compatibility with existing infrastructure is also a concern, including potential limitations in data, configuration, and transmission protocols. From a computing perspective, LLMs face challenges such as long response times, high bandwidth requirements, and data privacy concerns [79]. Deploying LLMs at the network edge is necessary to address these challenges. The staggering size of LLMs poses significant obstacles for mobile edge computing (MEC) systems. Balancing inference accuracy and memory usage is crucial when employing parameter sharing in LLMs. Furthermore, there are still numerous open research problems regarding the utilization of MEC systems to support LLMs. In terms of storage and caching [80], managing the computation and memory-intensive nature of LLMs is essential during loading and execution on edge servers. Core network latency and congestion can be problematic when offloading services for caching and inference, particularly due to the high number of service requests. Designing effective caching algorithms that consider the frequency of use for LLMs and user preferences is important. Dynamic cache structures based on service runtime configuration, such as batch size, add complexity to cache loading and eviction. Balancing the tradeoff between latency, energy consumption, and accuracy is a key consideration when managing cached models at edge servers.

## C. Collaborative Infrastructure for Mobile AIGC Networks

By asking ChatGPT the question "Integrating AI-generated content and mobile edge networks, please define mobile AIGC networks in one sentence," we can get the answer "Mobile AIGC networks are a fusion of AI-generated content and mobile edge networks, enabling rapid content creation, delivery, and processing at the network's edge for enhanced user experiences and reduced latency." (from Mar. 14 Version based on GPT-4) To support the pre-training, fine-tuning, and inference of the aforementioned models, substantial computation, communication, and storage resources are necessary. Consequently, to provide low-latency and personalized AIGC services, a collaborative cloud-edge-mobile AIGC framework shown in Fig. 6 is essential, requiring extensive cooperation among heterogeneous resource shareholders.

1) Cloud Computing: In mobile AIGC networks, cloud computing [81] represents a centralized infrastructure supplying remote server, storage, and database resources to support AIGC service lifecycle processes, including data collection, model training, fine-tuning, and inference. Cloud computing allows users to access AIGC services through the core network where these services are deployed, rather than building and maintaining physical infrastructure. Specifically, there are

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-12.jpg?height=1182&width=894&top_left_y=179&top_left_x=152)

Fig. 6: The collaborative cloud-edge-mobile infrastructure for mobile AIGC networks. The advantages and limitations of provisioning AIGC services in each layer are elaborated.

three primary delivery models in cloud computing: Infrastructure as a Service (IaaS), Platform as a Service (PaaS), and Software as a Service (SaaS). In mobile AIGC networks, IaaS providers offer access to virtualized AIGC computing resources such as servers, storage, and databases [23]. Additionally, PaaS provides a platform for developing and deploying AIGC applications and services. Lastly, SaaS delivers applications and services over the internet, enabling users to access generative AI models directly through a web browser or mobile application. In summary, cloud computing in mobile AIGC networks allows developers and users to harness the benefits of AI while reducing costs and mitigating challenges associated with constructing and maintaining physical infrastructure, playing a critical role in the development, deployment, and management of AIGC services.

2) Edge Computing: By providing computing and storage infrastructure at the edge of the core network [29], users can access AIGC services through radio access networks (RAN). Unlike the large-scale infrastructure of cloud computing, edge servers' limited resources often cannot support generative AI model training. However, edge servers can offer real-time finetuning and inference services that are less computationally and storage-intensive. By deploying edge computing at the network's periphery, users need not upload data through the

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-12.jpg?height=355&width=686&top_left_y=191&top_left_x=1183)

Fig. 7: The connections among AIGC services, wireless communication, mobile edge computing, and generative AI.

core network to cloud servers to request AIGC services. Consequently, reduced service latency, improved data protection, increased reliability, and decreased bandwidth consumption are benefits of AIGC services delivered via edge servers. Compared to exclusively delivering AIGC services through centralized cloud computing, location-aware AIGC services at the edge can significantly enhance user experience [82]. Furthermore, edge servers for local AIGC service delivery can be customized and personalized to meet user needs. Overall, edge computing enables users to access high-quality AIGC services with lower latency.

3) Mobile Computing: Device-to-device (D2D) mobile computing involves using mobile devices for the direct execution of AIGC services by users [18], [83]. On one hand, mobile devices can directly execute generative AI models and perform local AIGC inference tasks. While running generative AI models on devices demands significant computational resources and consumes mobile device energy, it reduces AIGC service latency and protects user privacy. On the other hand, mobile devices can offload AIGC services to edge or cloud servers operating over wireless connections, providing a flexible scheme for delivering AIGC services. However, offloading AIGC services to edge or cloud servers for execution necessitates stable network connectivity and increases service latency. Lastly, model compression and quantization must be considered to minimize the resources required for execution on mobile devices, as generative AI models are often large-scale.

Specifically, the connections among AIGC services, wireless communication, mobile edge computing, and generative AI are illustrated in Fig. 7

## D. Lessons Learned

1) Cloud-Edge Collaborative Training and Fine-tuning for Generative AI Models: To support AIGC services with required performance evaluated based on metrics discussed in Section III-A. cloud-edge collaborative pre-training and finetuning are envisioned to be promising approaches. On the one hand, cloud data centers can train generative AI models by using powerful computing and data resources. Pre-training in cloud data centers enables leveraging powerful computing and data resources and pre-training on large datasets, which can help models learn general features. However, AIGC services require significant communication and bandwidth resources, and thus raise privacy concerns, and may not be as effective
for fine-tuning on smaller more specific datasets. On the other hand, utilizing a large amount of user data in the edge network, the generative AI model can be fine-tuned to be more customized and personalized. The selection discusses the pros and cons of fine-tuning AIGC models on edge devices, including the utilization of user data available on edge devices, real-time interaction/response, and reduced privacy concerns, as well as limitations such as computing and storage resources and the need for specialized hardware and software.
2) Edge-Mobile Collaborative Inference for AIGC Services: In a mobile AIGC network, the user's location and mobility change over time [84]. Therefore, a large number of edge and mobile collaborations are required to complete the provision of AIGC inference services. Due to the different mobility of users, the AIGC services forwarded to the edge servers for processing are also dynamic. Several techniques can be leveraged to address the mobility issues in mobile AIGC networks, which include federated learning and distributed training to improve the efficiency of AIGC model updates, advanced DRL algorithms, and meta-learning techniques to optimize the AIGC provider selection strategy in response to changing network conditions, edge caching to deliver lowlatency content generation and computing services, and gathering user historical requests and profiles to provide personalized services. Therefore, dynamic resource allocation and task offloading decisions of AIGC applications are some of the challenges in deploying mobile AIGC networks, which we discuss in Section VI

## IV. How to Deploy AIGC at Mobile Edge NETWORKS: APPLICATIONS AND ADVANTAGES OF AIGC

This section introduces creative applications and advantages of AIGC services in the mobile edge network. Then, we provide four use cases of AIGC applications of mobile AIGC networks. Some examples of generative AI models are shown in Fig. 8 The applications elaborated in this section are summarized in Table $\square$

## A. Applications of Mobile AIGC Networks

1) AI-generated Texts: Recent advancements in Natural Language Generation (NLG) technology have led to AIgenerated text that is nearly indistinguishable from humanwritten text [11]. The availability of powerful open-source AI-generated text models, along with their reduced computing power requirements, has facilitated widespread adoption, particularly in mobile networks. The development of lightweight NLG models that can operate on resource-constrained devices, such as smartphones and IoT devices, while maintaining highperformance levels, has made AI-generated text an essential service in mobile AIGC networks [39].

One example of such a model is ALBERT (A Lite BERT), designed to enhance the efficiency of BERT (Bidirectional Encoder Representations from Transformers) while reducing its computational and memory requirements [119]. ALBERT is pre-trained on a vast corpus of text data and uses factorized embedding parameterization, cross-layer parameter sharing, and sentence-order prediction tasks to optimize BERT's performance while minimizing computational and memory demands. ALBERT has achieved performance levels comparable to BERT on various natural language processing tasks, such as question answering and sentiment analysis [12]. Its lighter model design makes it more suitable for deployment on edge devices with limited resources.

MobileBERT is another model designed for deployment on mobile and edge devices with minimal resources [120]. This more compact variant of the BERT model is pre-trained on the same amount of data as BERT but features a more computationally efficient design with fewer parameters. Quantization is employed to reduce the model's weight accuracy, further decreasing its processing requirements. MobileBERT is a highly efficient model compatible with various devices, including smartphones and IoT devices, and can be used in multiple mobile applications, such as personal assistants, chatbots, and text-to-speech systems [39]. Additionally, it can be employed in small-footprint cross-modal applications, such as image captioning, video captioning, and voice recognition. These AIgenerated text models offer significant advantages to mobile edge networks, enabling new applications and personalized user experiences in real time while preserving user privacy.

2) AI-generated Audio: AI-generated audio has gained prominence in mobile networks due to its potential to enhance user experience, and increase efficiency, security, personalization, cost-effectiveness, and accessibility [20]. For instance, AIGC-based speech synthesis and enhancement can improve call quality in mobile networks, while AIGC-based speech recognition and compression can optimize mobile networks by reducing the data required to transmit audio and automating tasks such as speech-to-text transcription. Voice biometrics powered by AI can bolster mobile network security by utilizing the user's voiceprint as a unique identifier for authentication [111]. AIGC-driven audio services, such as personalized music generation, can automate tasks and reduce network load, thereby cutting costs.

Audio Albert [49], a streamlined version of the BERT model adapted for self-supervised learning of audio representations, demonstrates competitive performance levels compared to other popular AI-generated audio models in various natural language processing tasks such as speech recognition, speaker identification, and music genre classification. In terms of latency, Audio Albert shows faster inference times than previous models, with a $20 \%$ reduction in average inference time on average, which can significantly improve response times in mobile edge networks. Additionally, Audio Albert's accuracy is comparable to BERT and achieves state-of-the-art results on several benchmarks. Furthermore, Audio Albert's model design is lighter than other models, making it suitable for deployment on edge devices with limited resources, improving computational efficiency while maintaining high-performance levels. Utilizing Audio Albert in mobile edge networks can provide several benefits, such as faster response times, reduced latency, and lower power consumption, making it a promising solution for AI-generated audio in mobile edge networks.

3) AI-generated Images: AI-generated images offer numerous applications in mobile networks, such as image en-
![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-14.jpg?height=452&width=894&top_left_y=148&top_left_x=171)

(b) DALLE-2

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-14.jpg?height=437&width=431&top_left_y=150&top_left_x=1075)

(c) Visual ChatGPT

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-14.jpg?height=279&width=187&top_left_y=256&top_left_x=1636)

(d) Point-E

Fig. 8: Generated images of different generative AI models, including Stable Diffusion (https://huggingface.co/spaces/ stabilityai/stable-diffusion), DALLE-2 (https://labs.openai.com/), Visual ChatGPT (https://huggingface.co/spaces/microsoft/ visual_chatgpt), Point-E (https://huggingface.co/spaces/openai/point-e), using the prompt "A photo of a green pumpkin".

TABLE II: Summary of State-of-the-art generative AI models.

| Application | Models | Network Archi- <br> tectures | Datasets | Evaluation Metrics |
| :---: | :---: | :---: | :---: | :---: |
| Text Generation | GPT-3 [85], GPT-4, BERT [86], <br> LaMDA [87], ChatGPT [12] | Transformer [88], <br> Diffusion Model | WebText, <br> BookCorpus <br> , Common <br> Crawl | BLEU [90], ROUGE <br> , Perplexity |
| Image Generation | StyleGAN [92], BigGANs [93], <br> StyleGANXL [94], DVD-GAN <br> , DALLE [8], DALLE2 [9], <br> CLIP [7], VisualGPT [96], VAE <br> , Energy-based GAN [72], <br> Flow-based models [74], Imagen <br> , diffusion probabilistic models <br> , DDPM [100], DDIM [101] | Diffusion Model, <br> GAN <br> VQ-VAE <br> Transformer 103$],$ <br> $88]$ | ImageNet [104], <br> CelebA 105$],$ <br> COCO 106 | FID [107], IS 108], <br> LPIPS [109] |
| Music Generation | MuseNet $\quad 110], \quad$ Jukedeck, <br> WaveNet [111], AudioLM 112] | Transformer, <br> RNN, CNN, <br> Diffusion Model | MIDI Dataset, <br> MAESTRO <br> $[113]$ | ABC-notation, Mu- <br> sic IS |
| Video Generation | ![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-14.jpg?height=126&width=516&top_left_y=1616&top_left_x=480) | Diffusion Model <br> $[1]$, GAN | Kinetics $[117$ | PSNR, SSIM |
| 3D Generation | NeRF [118] | Diffusion Model, <br> MLP | Synthetic and <br> real-world scenes | PSNR, SSIM, LPIPS |

hancement, image compression, image recognition, and textto-image generation [121]. Image enhancement can improve picture quality in low-light or noisy environments, while image compression decreases the data required to transmit images, enhancing overall efficiency. Various image recognition applications include object detection, facial recognition, and image search. Text-to-image generation enables the creation of images from textual descriptions for visual storytelling, advertising, and virtual reality/augmented reality (VR/AR) experiences [122]-125].

Make-a-Scene, a novel text-to-image generation model proposed in [126], leverages human priors to generate realistic images based on textual descriptions. The model consists of a text encoder, an image generator, and a prior human module trained on human-annotated data to incorporate common sense knowledge. In mobile networks, this model can be trained on a large dataset of images and textual descriptions to swiftly generate images in response to user requests, such as creating visual representations of road maps. This approach complements the techniques employed in [127] for generating images with specific attributes.

Furthermore, the Semi-Parametric Neural Image Synthesis (SPADE) method introduced in [127] generates new images from existing images and their associated attributes using a neural network architecture. This method produces highly realistic images conditioned on input attributes and can be employed for image-to-image translation, inpainting, and style transfer in mobile networks. The SPADE method shares similarities with the text-to-image generation approach in [126], where both techniques focus on generating high-quality, realistic images based on input data.

However, the development of AI-generated image tech-
nology also raises concerns around deep fake technology, which uses AI-based techniques to generate realistic photos, movies, or audio depicting nonexistent events or individuals, as discussed in [16]. Deep fakes can interfere with system performance and affect mobile user tasks, leading to ethical and legal concerns that require more study and legislation.

4) AI-generated Videos: AI-generated videos, like AIgenerated images, can be utilized in mobile networks for various applications, such as video compression, enhancement, summarization, and synthesis [95]. AI-generated videos offer several advantages over AI-generated images in mobile networks. They provide a more immersive and engaging user experience by dynamically conveying more information [128]. Moreover, AI-generated videos can be tailored to specific characteristics, such as style, resolution, or frame rate, to improve user experience or create videos for specific purposes, such as advertising, entertainment, or educational content [115]. Furthermore, AI-generated videos can generate new content from existing videos or other types of data, such as images, text, or audio, offering new storytelling methods [115].

Various models can be employed to achieve AI-generated videos in mobile networks. One such model is Imagen Video, presented in [13], which is a text-conditioned video generation system based on a cascade of video diffusion models. Imagen Video generates high-definition videos from text input using a base video generation model and an interleaved sequence of spatial and temporal video super-resolution models. The authors describe the process of scaling up the system as a high-definition text-to-video model, including design choices such as selecting fully-convolutional temporal and spatial super-resolution models at specific resolutions and opting for v-parameterization for diffusion models. They also apply progressive distillation with classifier-free guidance to video models for rapid, high-quality sampling [13], [115]. Imagen Video not only produces high-quality videos but also boasts a high level of controllability and world knowledge, enabling the generation of diverse videos and text animations in various artistic styles and with 3D object comprehension.

5) AI-generated $3 D$ : AI-generated $3 \mathrm{D}$ content is becoming increasingly promising for various wireless mobile network applications, including AR and VR [129], [130]. It also enhances network efficiency and reduces latency through optimal base station placement [131], [132]. Researchers have proposed several techniques for generating high-quality and diverse 3D content using deep learning (DL) models, some of which complement one another in terms of their applications and capabilities.

One such technique is the Latent-NeRF model, proposed in [133], which generates 3D shapes and textures from 2D images using the NeRF architecture. This model is highly versatile and can be used for various applications, such as $3 \mathrm{D}$ object reconstruction, 3D scene understanding, and 3D shape editing for wireless VR services. Another technique, the Latent Point Diffusion (LPD) model presented in [134], generates 3D shapes with fine-grained details while controlling the overall structure. LPD has been shown to create more diverse shapes than other state-of-the-art models, making it suitable for 3D shape synthesis, 3D shape completion, and 3D shape interpolation. The LPD model complements the latentNeRF approach by offering more diverse shapes and finer details.

Moreover, researchers in [135] proposed the Diffusion-SDF model, which generates 3D shapes from natural language descriptions. This model utilizes a combination of voxelized signed distance functions and diffusion-based generative models, producing high-quality 3D shapes with fine-grained details while controlling the overall structure. This technique accurately generates $3 \mathrm{D}$ shapes from natural language descriptions, making it useful for applications such as 3D shape synthesis, completion, and interpolation. It shares similarities with the Latent-NeRF and LPD models in terms of generating highquality $3 \mathrm{D}$ content [136].

## B. Advantages of Mobile AIGC

We then discuss several advantages of generative AI in mobile networks.

1) Efficiency: Generative AI models offer several efficiency benefits in mobile networks. As demonstrated in the applications of AI-generated text models like ALBERT [119] and MobileBERT [120], these models can automate the process of creating text, reducing the need for human labor and significantly boosting productivity [137]. Moreover, as shown in the applications of AI-generated audio models like Audio Albert [49], these models can be implemented at the edge of mobile networks [138], [139], allowing them to produce data locally on devices like smartphones and IoT sensors. This results in improved user experiences and reduced latency in mobile applications that rely on real-time data generation and processing [138].
2) Reconfigurability: The reconfigurability of AIGC in mobile networks is a significant advantage. As demonstrated in the ChatGPT application, AIGC can produce a vast array of content, which can be seamlessly adjusted to suit evolving network demands and user preferences [140]. Furthermore, as shown in the applications of AI-generated image models like Make-a-Scene [126] and SPADE [126], AIGC can contribute to reconfigurability in mobile networks by utilizing image and audio-generative models. These models can be trained to generate new visuals and auditory content based on specific parameters, such as user preferences or contextual information.
3) Accuracy: Employing generative AI models in mobile networks provides significant benefits in terms of accuracy, leading to more precise predictions and well-informed decision-making [114]. Similarly, AI-generated visuals and audio can be employed to improve the quality and accuracy of network-provided content, encompassing domains such as advertising, entertainment, and accessibility services. By using generative AI models, tailored and engaging content can be produced, resulting in a more impactful and personalized user experience. In the context of mobile networks, this can mean generating high-quality images or videos adapted to various devices and network conditions, improving the user perception of the provided services. By harnessing the power of generative AI models, mobile networks can offer more accurate and efficient services, ultimately fostering a superior
user experience and enabling innovative solutions tailored to the diverse needs of mobile users [47].
4) Scalability and Sustainability: Utilizing AIGC in mobile networks offers significant scalability and sustainability benefits [114]. AIGC can produce a wide range of content [13], enhancing mobile networks' overall scalability and sustainability in numerous ways. Specifically, AIGC facilitates scalability in mobile networks by reducing the reliance on human labor and resources. Furthermore, AIGC streamlines the entire content production process, encapsulating activities from initial capture to retouching, and from synergistic designer collaboration to large-scale production. This process efficiency leads to a substantial time saving, which not only results in diminished energy consumption, but also contributes to a reduced carbon footprint associated with maintaining physical storage infrastructures [141]. Despite the challenges associated with generative AI models, such as large model sizes and complex training processes, leveraging edge servers in mobile networks can help mitigate these issues by adopting an "AIGCas-a-Service" approach [138]. Users can interact with the system by submitting requests through their mobile devices and subsequently receiving computational results from edge servers. This strategy eliminates the need to deploy generative AI models on devices with constrained computing resources, optimizing overall efficiency and improving scalability and sustainability within the mobile network infrastructure [25].
5) Security and Privacy: AIGC can offer potential security and privacy advantages by embedding sensitive information within AI-generated content. This approach can serve as a form of steganography, a technique that conceals data within other types of data, making it difficult for unauthorized parties to detect the hidden information. However, it is essential to be aware of potential security and privacy risks associated with AIGC, such as adversarial attacks on AI models or the misuse of AI-generated content for malicious purposes, like deepfakes [16]. To ensure the secure and privacy-preserving use of AIGC in mobile networks, robust security measures and encryption techniques must be in place, along with ongoing research to counter potential threats [142].

## V. Case Studies of AIGC in Mobile NetworK

Many case studies have been done for achieving effective and efficient mobile AIGC networks as shown in Table. III. In this section, we review several representative cases, e.g., the AIGC service provider (ASP) selection, generative AIempowered traffic and driving simulation, AI-generated incentive mechanism, and blockchain-powered lifecycle management for AIGC.

## A. AI-Generated Incentive Mechanism

In this case study, we present the idea of using AI-generated optimization solutions with a focus on the use of diffusion models and their ability to optimize the utility function.

In today's world of advanced internet services, including the Metaverse, MR technology is essential for delivering captivating and immersive user experiences [162], [163]. Nevertheless, the restricted processing power of head-mounted displays (HMDs) used in MR environments poses a significant challenge to the implementation of these services. To tackle this problem, the researchers in [143] introduce an innovative information-sharing strategy that employs full-duplex deviceto-device semantic communication [164]. This method enables users to circumvent computationally demanding and redundant processes, such as producing AIGC in-view images for all MR participants. By allowing users to transmit generated content and semantic data derived from their view image to nearby users, these individuals can subsequently utilize the shared information to achieve spatial matching of computational outcomes within their view images. In their work, the authors of [143] primarily concentrate on developing a contract theoretic incentive mechanism to promote semantic information exchange among users. Their goal is to create an optimal contract that, while adhering to the utility threshold constraints of the semantic information provider, simultaneously maximizes the utility of the semantic information recipient. Consequently, they devised a diffusion model-based AI-generated contract algorithm [1], as illustrated in Fig. 9 .

Specifically, the researchers developed a cutting-edge algorithm for creating AI-generated incentive mechanisms [1], which tackle the challenge of utility maximization by devising optimal contract designs [143]. This approach is distinct from traditional neural network backpropagation algorithms or DRL methods, as it primarily focuses on enhancing contract design through iterative denoising of the initial distribution instead of optimizing model parameters. The policy for contract design is defined by the reverse process of a conditional diffusion model, linking environmental states to contract arrangements. The primary goal of this policy is to produce a deterministic contract design that maximizes the expected total reward over a series of time steps. To optimize system utility through contract design, the researchers in [143] create a contract quality network that associates an environment-contract pair with a value representing the expected total reward when an agent implements a particular contract design policy from the current state and adheres to it in the future. The optimal contract design policy maximizes the system's predicted cumulative utility. The researchers then carried out an extensive comparison between their suggested AI-powered contract algorithm and two DRL algorithms, specifically SAC and PPO. As illustrated in the training process in [143] (see Fig. 10), PPO requires more iteration steps to achieve convergence, while SAC converges more quickly but with a lower final reward value in comparison to the AI-driven contract algorithm.

The enhanced performance of the suggested AI-driven contract algorithm can be ascribed to two main aspects:

- Improved sampling quality: By configuring the diffusion step to 10 and applying multiple refinement steps, the diffusion models generate higher quality samples, mitigating the influence of uncertainty and augmenting sampling precision [114].
- Enhanced long-term dependence processing capability: Unlike conventional neural network generation models that take into account only the current time step input, the diffusion model creates samples with additional time steps through numerous refinement iterations, thereby

| Reference | System Model | Method Used |
| :---: | :---: | :---: |
| $[1]$ | A comprehensive tutorial on generative diffusion models in <br> various network optimization problems | Integration of diffusion models with DRL, incentive <br> design, semantic communications, and IoV networks |
| 143 | Users sharing information through full-duplex device-to-device <br> semantic communications | Diffusion model-based incentive mechanism genera- <br> tion to maximize the users' utilities |
| 144 | Selection of AIGC service providers (ASPs) capable of effec- <br> tively executing user tasks | Generative diffusion model for optimal decision gen- <br> eration in ASP selection problem |
| 145 | Distributed diffusion model where the user transmits the results <br> after several shared denoising steps to other users | A collaborative distributed diffusion-based AIGC <br> framework |
| 138 | Large-scale deployment of AaaS with 20 AIGC service <br> providers (ASPs) and 1000 edge users | Deep reinforcement learning (DRL)-enabled solution <br> to maximize a utility function |
| 146 | AIGC lifecycle management framework with three ESPs and <br> three producers, supported by the Draw Things application | Blockchain technology to protect the ownership and <br> copyright of AIGC, along with a reputation-based <br> service provider selection strategy |
| 147 | Deep generative model-empowered wireless network manage- <br> ment and use cases, e.g., network routing, resource allocation, <br> and network economics | Diffusion model to generate effective contracts for <br> incentivizing mobile AIGC services |
| [148 | Wireless sensing platform based on the 801.11 ac protocol with <br> a signal transmitter and five receivers | Multi-scale wireless perception for AIGC services |
| [149 | A user requests a specific number of images from a service <br> provider that is attacked by data poisoning, while diffusion <br> models provide the defense | Generative diffusion model-aided optimization to iden- <br> tify the optimal diffusion steps to minimize the total <br> energy cost |
| 150 | A multi-modality semantic-aware framework for generative <br> AI-enabled vehicular networks | A double deep Q-network-based approach to address <br> the resource allocation problem in generative AI- <br> enabled V2V communication |
| 151 | An integrated semantic communication and AIGC (ISCA) <br> framework for Metaverse services | Diffusion model-based joint resource allocation in <br> ISCA systems |
| 152 | A semantic communication framework based on You Only <br> Look Once (YOLO) to construct a virtual apple orchard | Semantic communications with generative diffusion <br> model-aided resource optimization |
| 153 | A foundation model caching and inference framework to <br> balance the tradeoff among inference latency, accuracy, and <br> resource consumption | Managing cached foundation models and user requests <br> during the provisioning of generative AI services |
| 80 | A framework of joint model caching and inference for man- <br> aging models and allocating resources | A least context algorithm for managing cached models <br> at edge servers |
| [154 | An autonomous driving architecture, where generative $\mathrm{AI}$ is <br> leveraged to synthesize conditioned traffic and driving data | A multi-task digital twin offloading model and a multi- <br> task enhanced auction-based mechanism |
| 155 | A framework that used mobile AIGC to drive Human Digital <br> Twin (HDT) applications, focusing on personalized healthcare <br> solutions | Using generative diffusion model for the resource <br> allocation in mobile AIGC-driven HDT system |
| $[156$ | The model combines Federated Learning (FL) with AIGC to <br> improve AIGC creation and privacy in wireless networks | Using FL techniques to fine-tune AIGC, yielding re- <br> duced communication cost and training latency |
| 157 | Exploring the application of Generative Artificial Intelligence <br> (GAI) in the physical layer of Integrated Sensing and Com- <br> munications (ISAC) systems | Using a diffusion model-based method for signal di- <br> rection estimation demonstrates GAI's efficacy in near- <br> field ISAC |
| 158 | GAI-aided Semantic Communication (SemCom) system that <br> uses multi-model prompts for accurate content decoding and <br> incorporates security measures | Using a diffusion model to ensure secure and accurate <br> message transmission |
| 159 | Using Pretrained Foundation Models (PFMs) and prompt engi- <br> neering to expand the applications of AIGC in edge networks | Using ChatGPT to train an effective prompt optimizer, <br> measuring its impact on user experience |
| 160 | Flexible-position multiple-input multiple-output (MIMO) sys- <br> tems | Using generative diffusion model to generate optimal <br> antenna trajectories to maximize system efficiency |
| 142 | A blockchain-aided semantic communication framework for <br> AIGC services in virtual transportation networks | A training-based targeted semantic attack scheme and <br> counters it with a blockchain and zero-knowledge <br> proof-based defense mechanism |
| 142 | A blockchain-aided semantic communication framework for <br> AIGC services in virtual transportation networks | A training-based targeted semantic attack scheme and <br> counters it with a blockchain and zero-knowledge <br> proof-based defense mechanism |
| [161 | A framework that uses wireless perception to guide generative <br> AI in producing digital content | A Sequential Multi-Scale Perception algorithm for <br> user skeleton prediction and a diffusion model-based <br> approach to generate an optimal pricing strategy |

TABLE III: Key Literature Considering AIGC within Wireless Network.

bolstering its long-term dependence processing capability $[121]$.

As demonstrated in Fig. 10, the authors in [143] examine the optimal contract design capacities of the trained models. For a specific environmental state, the AI-driven contract algorithm provides a contract design that attains a utility value of 189.1,

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-18.jpg?height=407&width=1805&top_left_y=187&top_left_x=152)

Fig. 9: System model of contract design in semantic information sharing network, and the AI-generated contract algorithm. The diffusion models generate different optimal contract designs under different environmental variables.

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-18.jpg?height=515&width=658&top_left_y=802&top_left_x=251)

(a) Training process, with diffusion step $N=10$ [143].

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-18.jpg?height=456&width=639&top_left_y=1412&top_left_x=274)

(b) The designed contracts.

Fig. 10: The effect of different incentive design schemes, e.g., PPO, SAC, and AI-generated contract [143].

markedly outperforming SAC's 185.9 and PPO's 184.3. These results highlight the practical advantages of the proposed AI-based contract algorithm in contrast to traditional DRL techniques.

Lesson Learned: The case study in this research highlights the potential of AI-generated optimization solutions, particularly diffusion models, for addressing complex utility maximization problems within incentive mechanism design. The authors in [143] present an innovative approach that employs full-duplex device-to-device semantic communication for information-sharing in mixed reality environments, overcoming the limitations of HMDs. The diffusion model-based
AI-generated contract algorithm proposed in this study demonstrates superior performance compared to traditional DRL algorithms, such as SAC and PPO. The superior performance of the AI-generated contract algorithm can be attributed to improved sampling quality and enhanced long-term dependence processing capability. This study underscores the effectiveness of employing AI-generated optimization solutions in complex, high-dimensional environments, particularly in the context of incentive mechanism design. Some promising directions for future research include:

- Expanding the application of diffusion models: Investigate the application of diffusion models in other domains, such as finance, healthcare, transportation, and logistics, where complex utility maximization problems often arise.
- Developing novel incentive mechanisms: Explore the development of new incentive mechanisms that combine AI-generated optimization solutions with other approaches, such as game theory or multi-agent reinforcement learning, to create even more effective incentive designs.
- Exploring the role of human-AI collaboration: Investigate how AI-generated optimization solutions can be combined with human decision-making to create hybrid incentive mechanisms that capitalize on the strengths of both human intuition and AI-driven optimization.


## B. AIGC Service Provider Selection

The integration of generative AI models within wireless networks offers significant potential, as these state-of-theart technologies have exhibited exceptional capabilities in generating a wide range of high-quality content. By harnessing the power of artificial intelligence, generative AI models can astutely analyze user inputs and produce tailored, contextually relevant content in real-time 114]. This stands to considerably enhance user experience and foster the creation of innovative applications across various domains, such as entertainment, education, and communication. Nonetheless, the deployment and application of these advanced models give rise to challenges, including extensive model sizes, complex training processes, and resource constraints. Consequently, deploying large-scale AI models on every network edge device poses considerable difficulties.

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-19.jpg?height=651&width=875&top_left_y=192&top_left_x=167)

Fig. 11: The system model of AIGC service provider selection. Different ASPs performing user tasks can bring different results and different user utilities. Considering that different mobile users have different task requirements and different ASP's AI models have different capabilities and computation capacities, a proper ASP selection algorithm is needed to maximize the total utilities of network users.

To address this challenge, the authors in [138] introduce the "AIGC-as-a-service" architecture. This approach entails ASPs deploying AI models on edge servers, which facilitates the provision of instantaneous services to users via wireless networks, thereby ensuring a more convenient and adaptable experience. By enabling users to effortlessly access and engage with AIGC, the proposed solution minimizes latency and resource consumption. Consequently, edge-based AIGC-asa-service holds the potential to transform the creation and delivery of AIGC across wireless networks.

However, one problem is that the effectiveness of ASP in meeting user needs displays significant variability due to a variety of factors. Certain ASPs may concentrate on generating specific content types, while others boast more extensive content generation capabilities. For instance, some providers may specialize in producing particular content categories, whereas others offer a wider range of content generation options. Moreover, several ASPs may have access to advanced computing and communication resources, empowering them to develop and deploy more sophisticated generative AI models within the mobile network. As depicted in Fig. 11, users uploading images and requirement texts to different ASPs encounter diverse results owing to the discrepancies in models employed. For example, a user attempting to add snow to grass in an image may experience varying outcomes depending on the ASP chosen.

With a large number of mobile users and increasing demand for accessing requests, it is crucial to analyze and select ASPs with the necessary capability, skill, and resources to offer high-quality AIGC services. This requires a rigorous selection process considering the provider's generative AI model capabilities and computation resources. By selecting a provider with the appropriate abilities and resources, organizations can ensure that they have effective AIGC services to increase the $\mathrm{QoE}$ for mobile users. Motivated by the aforementioned reasons, the authors in [138] examine the viability of largescale deployment of AIGC-as-a-Service in wireless edge networks. Specifically, in the ASP selection problem, which can be framed as a resource-constrained task assignment problem, the system consists of a series of sequential user tasks, a set of available ASPs, and the unique utility function for each ASP. The objective is to find an assignment of tasks to ASPs, such that the overall utility is maximized. Note that the utility of the task assigned to the ASP is a function of the required resource. Without loss of generality, the authors in [138] consider that is in the form of the diffusion step of the diffusion model, which is positively correlated to the energy cost. The reason is that each step of the diffusion model has energy consumption as it involves running a neural network to remove Gaussian noise. Finally, the total availability of resources for each ASP is taken into account to ensure that the resource constraints are satisfied.

In this formulation of AIGC service provisioning, the resource constraints are incorporated through the resource constraint, which specifies the limitations on the available resources. Note that failing to satisfy the resource constraint can result in the crash of ASP, causing the termination and restart of its running tasks.

Several baseline policies are used for comparison:

- Random Allocation Policy. This strategy distributes tasks to ASPs in a haphazard manner, without accounting for available resources, task duration, or any restrictions. The random allocation serves as a minimum benchmark for evaluating scheduling efficiency.
- Round-Robin Policy. The round-robin policy allocates tasks to ASPs sequentially in a repeated pattern. This approach can generate effective schedules when tasks are evenly distributed. However, its performance may be suboptimal when there are significant disparities among them.
- Crash-Avoid Policy. The crash-avoid policy prioritizes ASPs with greater available resources when assigning tasks. The goal is to prevent overburdening and maintain system stability.
- Upper Bound Policy. In this hypothetical scenario, the scheduler has complete knowledge of the utility each ASP offers to every user before task distribution. The omniscient allocation strategy sets an upper limit on the performance of user-centric services by allocating tasks to ASPs with the highest utility and avoiding system failures. However, this approach relies on prior information about the unknown utility function, which is unrealistic in practice.

The authors in [138] employed a Deep Reinforcement Learning (DRL) technique to optimize Application Service Provider (ASP) selection. In particular, they implemented the Soft Actor-Critic (SAC) method, which alternates between evaluating and improving the policy. Unlike traditional actor-critic frameworks, the SAC approach maximizes a balance between expected returns and entropy, allowing it to optimize both

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-20.jpg?height=567&width=740&top_left_y=215&top_left_x=215)

Fig. 12: The cumulative rewards under different ASP selection algorithms [138]. DRL-based algorithms can outperform multiple baseline policies, i.e., overloading-avoidance, random, and round-robin, and approximate the optimal policy.

exploitation and exploration for efficient decision-making in dynamic ASP selection scenarios. To conduct the simulation, the authors consider 20 ASPs and 1000 edge users. Each ASP offered AaaS with a maximum resource capacity, measured by total diffusion timesteps in a given time frame, varying randomly between 600 and 1,500. Each user submits multiple AIGC task requests to ASPs at varying times. These requests detailed the necessary AIGC resources in terms of diffusion timesteps, randomly set between 100 and 250. Task arrivals from users adhered to a Poisson distribution, with a rate of 0.288 requests per hour over a 288 -hour duration, amounting to 1,000 tasks in total. As shown in Fig. 12. simulation results indicate that the proposed DRL-based algorithm outperforms three benchmark policies, i.e., overloading-avoidance, random, and round-robin, by producing higher-quality content for users and achieving fewer crashed tasks. Lesson Learned: The lesson learned from this study is that the proper selection of ASPs is crucial for maximizing the total utilities of network users and enhancing their experience. The authors in [138] introduced a DRL-based algorithm for ASP selection, which outperforms other baseline policies, such as overloadingavoidance, random, and round-robin. By leveraging the SAC approach, the algorithm strikes a balance between exploitation and exploration in decision-making for dynamic ASP selection scenarios. Consequently, this method can provide higherquality content for users and lead to fewer crashed tasks, ultimately improving the quality of service in wireless edge networks. To further enhance research in the area of AIGC service provider selection, future studies could have:

- Investigate the integration of FL and distributed training methods to improve the efficiency of generative AI model updates and reduce the communication overhead among ASPs.
- Explore advanced DRL algorithms and meta-learning techniques to adaptively adjust the ASP selection strategy in response to changing network conditions and user requirements.

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-20.jpg?height=591&width=878&top_left_y=192&top_left_x=1079)

Fig. 13: Generative AI-empowered simulations for autonomous driving in vehicular Metaverse, which consists of $\mathrm{AVs}$, virtual simulators, and roadside units.

- Assess the impact of real-world constraints, such as network latency, data privacy, and security concerns, on the ASP selection process and devise strategies to address these challenges.
- Develop multi-objective optimization techniques for ASP selection that consider additional factors, such as energy consumption, cost, and the trade-off between content quality and computational resources.


## C. Generative AI-empowered Traffic and Driving Simulation

In autonomous driving systems, traffic and driving simulation can affect the performance of connected autonomous vehicles (AVs). Existing simulation platforms are established based on historical road data and real-time traffic information. However, these data collection processes are difficult and costly, which hinders the development of fully automated transportation systems. Fortunately, generative AI-empowered simulations can largely reduce the cost of data collection and labeling by synthesizing traffic and driving data via generative AI models. Therefore, as illustrated in Fig. 13, the authors in [154] design a specialized generative AI model, namely TSDreambooth, for conditional traffic sign generation in the proposed vehicular mixed reality Metaverse architecture. In detail, TSDreambooth is a variation of stable diffusion [165] fine-tuned based on the Belgium traffic sign (BelgiumTS) dataset [166]. The performance of TSDreambooth is validated via the pre-trained traffic sign classification model as generative scores. In addition, the newly generated datasets are leveraged to improve the performance of original traffic sign classification models.

In the vehicular Metaverse, connected $\mathrm{AV}$ s, roadside units, and virtual simulators can develop simulation platforms in the virtual space collaboratively. Specifically, AVs maintain their representations in the virtual space via digital twin (DT) technologies. Therefore, AVs need to continuously generate multiple DT tasks and execute them to update the representations. To offload these DT tasks to roadside units for remote execution in real-time, $\mathrm{AV}$ s need to pay for the

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-21.jpg?height=569&width=653&top_left_y=217&top_left_x=281)

Fig. 14: Performance evaluation of the MTEPViSA under different sizes of the market.

communication and computing resources of roadside units. Therefore, to provide fine-grained incentives for RSUs in executing DT tasks with heterogeneous resource demands and various required deadlines, the authors in [154] propose a multi-task enhanced physical-virtual synchronization auctionbased mechanism, namely MTEPViSA, to determine and price the resources of RSUs. There are two stage of this mechanism the online submarket for provisioning DT services and the offline submarket for provisioning traffic and driving simulation services. In the online simulation submarket, the multi-task DT scoring rule is proposed to resolve the externalities from the offline submarket. In the meanwhile, the price scaling factor is leveraged to reduce the effect of asymmetric information among driving simulators and traffic simulators in the offline submarket. The simulation experiments are performed in a vehicular Metaverse system with 30 AVs, 30 virtual traffic simulators, 1 virtual driving simulator, and 1 RSU. The experimental results demonstrate that the proposed mechanism can improve $150 \%$ social surplus compared with other baseline mechanisms. Finally, they develop a simulation testbed of generative AI-empowered simulation systems in the vehicular Metaverse.

The vehicular mixed-reality (MR) Metaverse simulation environment was constructed employing a 3D model representing several city blocks within New York City. Geopipe, Inc. developed this model by leveraging artificial intelligence to generate a digital replica based on photographs taken throughout the city. The simulation encompasses an autonomous vehicle navigating a road, accompanied by strategically positioned highway advertisements. Eye-tracking data were gathered from human participants immersed in the simulation, utilizing the HMD Eyes addon provided by Pupil Labs. Subsequent to the simulation, participants completed a survey aimed at evaluating their subjective level of interest in each simulated scenario. As the experimental results shown in Fig. 14, According to the study, as the number of AVs continues to increase, the supply and demand mechanisms in the market are changing. Therefore, to improve market efficiency and total surplus, some mechanisms need to be adopted to coordinate supply and demand. We investigate the market mechanism and propose a mechanism based on AIGC technology to enhance market efficiency. Compared with the existing Physical-virtual Synchronization auction (PViSA) and Enhanced Physical-virtual Synchronization auction (EPViSA) mechanisms [167], [168], the AIGC-empowered mechanism can double the total surplus under different numbers of $\mathrm{AVs}$.

Lesson Learned: This case study on generative AIempowered autonomous driving opens a new paradigm for the vehicular Metaverse, where data and resources can be utilized more efficiently. The authors demonstrate the potential of generative AI models in synthesizing traffic and driving data to reduce the cost of data collection and labeling. The proposed MTEPViSA mechanism also provides a solution to determine and price the resources of roadside units for remote execution of digital twin tasks, improving market efficiency and total surplus. However, there are still several open issues that need to be addressed in this field. Firstly, it is necessary to investigate the potential negative impacts of generative AI models in synthesizing traffic and driving data, such as biases and inaccuracies. Secondly, more research is needed to develop robust and trustworthy mechanisms for determining and pricing the resources of RSUs to ensure fair and efficient allocation of resources. Thirdly, the proposed mechanism needs to be tested and evaluated in more complex and varied scenarios to ensure its scalability and applicability in real-world situations.

## D. Blockchain-Powered Lifecycle Management for AIGenerated Content Products

This case study delves into the application of a blockchainbased framework for managing the lifecycle of AIGC products within edge networks. The framework, proposed by the authors in [146], addresses concerns related to stakeholders, the blockchain platform, and on-chain mechanisms. We explore the roles and interactions of the stakeholders, discuss the blockchain platform's functions, and elaborate on the framework's on-chain mechanisms. Within edge networks, the AIGC product lifecycle encompasses four main stakeholders: content creators, Edge Service Providers (ESPs), end-users, and adversaries. The following describes their roles and interplay within the system:

- Producers: Initiate the AIGC product lifecycle by proposing prompts for ESPs to generate content. They retain ownership rights and can publish and sell the generated products.
- ESPs: Possess the resources to generate content for producers, charging fees based on the time and computing power used for the tasks.
- Consumers: View and potentially purchase AIGC products, participating in multiple trading transactions throughout the product lifecycle.
- Attackers: Seek to disrupt normal operations of AIGC products for profit through ownership tampering and plagiarism.

Considering the roles of these stakeholders, the blockchain platform fulfills two primary functions: providing a traceable
and immutable ledger and supporting on-chain mechanisms. Transactions are recorded in the ledger and validated by full nodes using a consensus mechanism, ensuring security and traceability. ESPs act as full nodes, while producers and consumers serve as clients.

To address the concerns arising from stakeholder interactions, the framework employs three on-chain mechanisms [146]:

- Proof-of-AIGC: A mechanism that defends against plagiarism by registering AIGC products on the blockchain. It comprises two phases: proof generation and challenge.
- Incentive Mechanism: Safeguards the exchange of funds and AIGC ownership using Hashed Timelock Contracts (HTLCs).
- Reputation-based ESP Selection: Efficiently schedules AIGC generation tasks among ESPs based on their reputation scores.

The Proof-of-AIGC mechanism plays a vital role in maintaining the integrity of AIGC products. It encompasses two stages: proof generation and challenge. The objective of proof generation is to record AIGC products on the blockchain, while the challenge phase allows content creators to raise objections against any on-chain AIGC product they deem infringing upon their creations. If the challenge is successful, the duplicate product can be removed from the registry, thus protecting the original creator's intellectual property rights.

To further strengthen the security of the AIGC ecosystem, a pledged deposit is necessary to initiate a challenge, preventing arbitrary challenges that could burden the blockchain. This process comprises four steps: fetching the proofs, verifying the challenger's identity, measuring the similarity between the original product and the duplicate, and checking the results.

The AIGC economic system necessitates an incentive mechanism to motivate stakeholders and ensure legitimate exchanges of funds and ownership. The Incentive Mechanism rewards ESPs for maintaining the ledger and providing blockchain services. There are no transaction fees, and block generators follow a first-come-first-serve strategy. A two-way guarantee protocol using Hash Time Lock (HTL) is designed to build mutual trust and facilitate AIGC circulation during both the generation and trading phases.

The Proof-of-AIGC mechanism tackles issues like ownership manipulation and AIGC plagiarism, while the incentive mechanism ensures compliance with pre-established contracts. Furthermore, a reputation-based ESP selection accommodates ESP heterogeneity, which is crucial for efficient AIGC lifecycle management. Specifically, within the AIGC lifecycle management architecture, producers can concurrently interact with multiple heterogeneous ESPs, necessitating the identification of a trustworthy ESP for a specific task. Conventional approaches involve selecting the most familiar ESP to minimize potential risks, which may result in unbalanced workload distribution and increased service latency among ESPs. To address this challenge, a reputation-based ESP selection strategy is incorporated into the framework. This strategy ranks all accessible ESPs according to their reputation, which is computed using Multi-weight Subjective Logic (MWSL). The primary objectives are to assist producers in choosing the most

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-22.jpg?height=642&width=810&top_left_y=178&top_left_x=1118)

Fig. 15: The reputation trends of three ESPs (from the perspective of a random producer) [146.

reliable ESP, distribute the workload evenly across multiple ESPs, and motivate ESPs to accomplish tasks promptly and honestly, as a negative reputation impacts their earnings.

Producers identify suitable ESPs by computing the reputation of all potential ESPs, ranking them based on their current reputation, and allocating the AIGC generation task to the ESP with the highest standing. In MWSL, the concept of "opinion" serves as the fundamental element for reputation calculation. Local opinions represent the assessments of a specific producer who has directly interacted with the ESPs, while recommended opinions are derived from other producers who have also engaged with the ESPs. To mitigate the effect of subjectivity, an overall opinion is generated for each producer by averaging all the acquired recommended opinions. As producers possess varying degrees of familiarity with ESPs, the weight of their recommended opinions differs. Reputation is determined by combining a producer's local opinion with the overall opinion. The reputation scheme accomplishes its design objectives by quantifying the trustworthiness of ESPs, aiding producers in selecting the most dependable ESP, reducing service bottlenecks, and incentivizing ESPs to deliver high-quality AIGC services to maximize their profits.

A demonstration of the AIGC lifecycle management framework is conducted to verify the proposed reputation-based ESP selection approach [146]. The experimental setup comprises three ESPs and three producers, with the AIGC services facilitated by the Draw Things application. Several parameters are configured, and producers can employ the Softmax function to ascertain the probability of choosing each ESP. The reputation trends of the three ESPs are shown in Fig. 15, with ESP1 attaining the highest rank and remaining stable owing to its superior service quality. When ESP1 deliberately postpones AIGC services, its reputation declines sharply, while the reputations of ESP2 and ESP3 continue to rise. The proposed reputation strategy effectively measures the trustworthiness of ESPs, enabling producers to effortlessly discern the most reliable ESPs and motivating ESPs to operate with integrity. In reality, the dynamics of ESP selection would become more

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-23.jpg?height=591&width=824&top_left_y=179&top_left_x=192)

Fig. 16: The total number of assigned tasks of three ESPs 146 ].

complex with an increase in the number of ESPs and producers. This underlines the potential challenges and importance of effective reputation management strategies in such expanded scenarios. The reputation-based selection method's robustness and scalability in a larger network is a subject for future work. The workload of ESPs under different ESP selection methods is also demonstrated in Fig. 16. Traditional methods lead to uneven workloads and extended service latencies. Conversely, the proposed reputation-based method effectively balances the workload among ESPs. This is achieved by enabling producers to quantitatively assess the trustworthiness of ESPs without solely relying on their experiential judgment. The effectiveness of this approach in a network with a larger number of ESPs is an aspect that invites further exploration.

Lesson Learned: The case study on blockchain-powered lifecycle management for AI-generated content products highlights the potential of a blockchain-based framework in addressing key concerns like stakeholder interactions, platform functionality, and on-chain mechanisms. The primary lessons learned emphasize the importance of defining clear stakeholder roles, implementing robust mechanisms such as Proof-ofAIGC and Incentive Mechanism to ensure system integrity, and employing a reputation-based ESP selection scheme to balance workload and encourage honest performance. These insights collectively contribute to the effective management of the AIGC product lifecycle within edge networks. Future research in blockchain-powered lifecycle management for AIgenerated content products can explore several promising directions:

- Enhancing the efficiency and scalability of the blockchain platform to handle an increased number of transactions and support a growing AIGC ecosystem might be critical.
- Refining the reputation-based ESP selection scheme to account for more sophisticated factors, such as task complexity, completion time, and user feedback, could lead to more accurate and dynamic trustworthiness evaluations.
- Incorporating privacy-preserving techniques to protect sensitive data in AIGC products and user information without compromising the transparency and traceability of blockchain technology would be valuable.


## VI. Implementation ChallengeS in Mobile AIGC NETWORKS

When providing AIGC services, a significant amount of computational and storage resources are required to run the generative AI model. These computation and storage-intensive services pose new challenges to existing mobile edge computing infrastructure. As discussed in Section III-C a cloudedge-mobile collaborative computing architecture can be implemented to provide AIGC services. However, several critical implementation challenges must be addressed to improve resource utilization and the user experience.

## A. Edge Resource Allocation

AIGC service provisioning based on edge intelligence is computationally and communication-intensive for resourceconstrained edge servers and mobile devices [169], [170]. Specifically, AIGC users send service allocation requests to edge services. Upon receiving these AIGC requests, edge servers perform the AIGC tasks and deliver the output to users [171. During this AIGC service provisioning interaction, model accuracy and resource consumption are the most common metrics. Consequently, significant efforts are being made to coordinate mobile devices and edge servers for deploying generative AI at mobile edge networks. As summarized in Table IV, several Key Performance Indicators (KPIs) for edge resource allocation in AIGC networks are presented below. Here are several KPIs for edge resource allocation in AIGC networks.

- Model accuracy: In a resource-constrained edge computing network, a key issue when allocating edge resources is optimizing the accuracy of AI services while fully utilizing network resources [179]. Besides objective image recognition and classification tasks, AI models are also based on the content's degree of personalization and adaptation. Thus, optimizing AIGC content networks may be more complex than traditional optimization since personalization and customization make evaluating model accuracy more unpredictable.
- Bandwidth utilization: While providing AIGC services, the edge server must maximize its channel utilization to ensure reliable service in a high-density edge network. To allocate its bandwidth resources more efficiently, the edge server must control channel access to reduce interference between user requests and maximize the quality of its AIGC service to attract more users.
- Edge resource consumption: Deploying AIGC services in edge networks requires computationally intensive AI training and inference tasks that consume substantial resources. Due to the heterogeneous nature of edge devices, edge services consume resources in generating appropriate AIGC while processing users' requests [180]. Deployment of AIGC services necessitates continuous iteration to meet actual user needs, as generation results of generative AI models are typically unstable. This

TABLE IV: Summary of scenarios, problems, benefits/challenges, and mathematical tools of edge resource allocation.

| Ref. | Scenarios | Performance <br> Metrics/Decision <br> Variables | Benefits/Challenges | Mathematical Tools |
| :---: | :---: | :---: | :---: | :---: |
| $[172$ | Adaptive control for <br> distributed edge learning | Model loss/Steps of local <br> updates, the total number <br> of iterations | Provisioning AIGC services <br> in resource-constrained <br> edge environments | Control theory |
| $\|173\|$ | Geo-distributed ML | Execution time/Selective <br> barrier, mirror clock | Provisioning Localized <br> AIGC services | Convergence analysis |
| $\mid 174$ | AI service placement in <br> mobile edge intelligence | Total time and energy <br> consumption/Service <br> placement decision, local <br> CPU frequencies, uplink <br> bandwidth, edge CPU <br> frequency | Fully utilize scarce wireless <br> spectrum and edge <br> computing resources in <br> provisioning AIGC services | ADMM |
| $[175]$ | Joint model training and <br> task inference | Energy consumption and <br> execution latency/Model <br> download decision and task <br> splitting ratio | Integrated fine-tuning and <br> inference for generative AI <br> models with heterogeneous <br> computing resources | ADMM |
| $\mid 176$ | Serving edge DNN <br> inference for multiple <br> applications and multiple <br> models | Inference accuracy, latency, <br> resource cost/Application <br> configuration, DNN model <br> selection, and edge <br> resources | Provision rich AIGC <br> services for long-term <br> utility maximization | Regularization-based online <br> optimization |
| [177 | Multi-user collaborative <br> DNN partitioning | Execution <br> latency/Partitioning, <br> computation resources | Providing insights for <br> partitioning generative $\mathrm{AI}$ <br> models under edge-mobile <br> collaboration | Iterative alternating <br> optimization |
| [178 | Hierarchical federated edge <br> learning | Data convergence and <br> revenue/Cluster selection <br> and payment | Provisioning <br> privacy-preserving AIGC <br> services in edge networks | Evolutionary game and <br> auction |

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-24.jpg?height=604&width=879&top_left_y=1362&top_left_x=165)

Fig. 17: Dynamic AIGC application configuration and generative AI model compression for serving AIGC services in mobile AIGC networks.

constant AIGC service provisioning at edge servers leads to significant resource consumption.

Obtaining a balance between model accuracy and resource consumption can be challenging in resource-constrained edge computing networks. One potential strategy is to adjust the trade-off between model accuracy and resource consumption according to the needs of the users. For example, in some cases, a lower level of model accuracy may be acceptable if it results in faster response times or lower resource consumption. Another approach is to use transfer learning, which involves training an existing model on new data to improve accuracy while requiring fewer computational resources. Model compression techniques can also be used to reduce the size of the AI model without significantly impacting accuracy. However, it is important to note that these techniques may not be applicable in all scenarios, as personalization and customization can make evaluating model accuracy more unpredictable. Deployment of AIGC services necessitates continuous iteration to meet actual user needs, as generation results of generative AI models are typically unstable. Due to the heterogeneous nature of edge devices, edge services consume resources in generating appropriate AIGC while processing users' requests. This constant AIGC service provisioning at edge servers leads to significant resource consumption.

To provide intelligent applications at mobile edge networks, considerable effort should focus on the relationship between model accuracy, networking, communication, and computation resources at the edge. Simultaneously, offering AIGC services is challenging due to the dynamic network environment and user requirements at mobile edge networks. The authors in [173] propose a threshold-based approach for reducing traffic at edge networks during collaborative learning. By considering computation resources, the authors in [172] examine the distributed ML problem under communication, computation, storage, and privacy constraints. Based on the theoretical results obtained from the distributed gradient descent convergence rate, they propose an adaptive control algorithm for distributed edge learning to balance the trade-off between local updates and global parameter aggregations. The experimental results
demonstrate the effectiveness of their algorithm under various system settings and data distributions.

Generative AI models often require frequent fine-tuning and retraining for newly generated data and dynamic requests in non-stationary mobile edge networks [181]. Due to limited storage resources at edge servers and the different customization demands of AIGC providers, the AIGC service placement problem is investigated in [174]. To minimize total time and energy consumption in edge AI systems, the AI service placement and resource allocation problem is formulated as an MINLP. In the optimization problem, AI service placement and channel allocation are discrete decision variables, while device and edge frequencies are continuous variables. However, solving this problem is not trivial, particularly in largescale network environments. Thus, the authors propose an alternating direction method of multipliers (ADMM) to reduce the complexity of solving this problem. The experimental results demonstrate that this method achieves near-optimal system performance while the computational complexity grows linearly as the number of users increases. Moreover, when edge intelligence systems jointly consider AI model training and inference [175], the ADMM method can optimize edge resources. Additionally, the authors [176] explore how to serve multiple AI applications and AI models at the edge. They propose EdgeAdapter, as illustrated in Fig. 17, to balance the triple trade-off between inference accuracy, latency, and resource consumption. To provide inference services with long-term profit maximization, they first analyze the problem as an NP-hard problem and then solve it with a regularizationbased online algorithm.

In mobile AIGC networks, an effective architecture for providing AIGC services is to partition a large generative $\mathrm{AI}$ model into multiple smaller models for local execution [32]. In [177], the authors consider a multi-user scenario with massive IoT [182] devices that cooperate to support an intelligent application collaboratively. Although partitioning large ML models and distributing smaller models to mobile devices for collaborative execution is feasible, the model distribution and result aggregation might incur extra latency during model training and inference. Additionally, the formulated optimization problem is complex due to its numerous constraints and vast solution space. To address these issues, the authors propose an alternative iterative optimization to obtain solutions in polynomial time. Furthermore, AIGC services allow users to input their preferences into generative AI models. Therefore, to preserve user privacy among multiple users during collaborative model training and inference [183], the authors in [178] investigate the communication efficiency issues of decentralized edge intelligence enabled by FL. In the FL network, thousands of mobile devices participate in model training. However, selecting appropriate cluster heads for aggregating intermediate models can be challenging. Decentralized learning approaches can improve reliability while sacrificing some communication performance, unlike centralized learning with a global controller. A two-stage approach can be adopted in decentralized learning scenarios to improve the participation rate. In this approach, evolutionary game-based allocation can be used for cluster head selection, and DL-based auction

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-25.jpg?height=607&width=897&top_left_y=190&top_left_x=1075)

Fig. 18: Model partitioning in mobile AIGC networks. The generative AI models of mobile devices can be split and full or partial of them can be offloaded to edge servers for remote execution.

effectively rewards model owners.

## B. Task and Computation Offloading

In general, executing generative AI models that generate creative and valuable content necessitates substantial computational resources, which is impractical for mobile devices with limited resources [25], [190]. Offering high-quality and low-latency AIGC services is challenging for mobile devices with low processing power and limited battery life. Fortunately, AIGC users can offload the tasks and computations of generative AI models over the RAN to edge servers located in proximity to the users. This alleviates the computational burden on mobile devices.

As listed in Table V. several KPIs are specifically relevant to computation offloading in mobile AIGC networks:

- Service latency: Service latency refers to the delay associated with data input and retrieval as well as the model inference computations that users perform to generate AIGC [191]. By offloading AIGC tasks from mobile devices, such as fine-tuning and inference, to edge servers for execution, the total latency in mobile AIGC networks can be reduced. Unlike local execution of the generative AI model, offloading AI tasks to the edge server for execution introduces additional latency when transmitting personalized instructions and downloading AIGC content.
- Reliability: Reliability evaluates users' success rate in obtaining personalized data accurately. On the one hand, when connecting to the edge server, users may experience difficulty uploading the requested data to edge servers or downloading the results from servers due to dynamic channel conditions and wireless network instability. On the other hand, the content generated by the generative AI model may not fully meet the needs of AIGC users in terms of personalization and customization features. Unsuccessful content reception and invalid content affect the AIGC network's reliability.

TABLE V: Summary of scenarios, problems, benefits/challenges, and mathematical tools of task and computation offloading.

| Ref. | Scenarios | Performance <br> Metrics/Decision variables | Benefits/Challenges | Mathematical Tools |
| :---: | :---: | :---: | :---: | :---: |
| [184 | Edge intelligence in IoT | Processing delay/Task <br> offloading decisions | Offload AIGC tasks for <br> improving inference <br> accuracy | Optimization theory |
| [185 | Intelligent IoT applications | Processing time/Offloading <br> decisions | Support on-demand <br> changes for AIGC <br> applications | Random forest regression |
| $\mid 32$ | Collaborative intelligence <br> between the cloud and <br> mobile edge | Latency and energy <br> consumption/DNN <br> computation partitioning | Cloud and mobile edge <br> collaborative intelligence <br> for generative AI models | Greedy algorithm |
| $\|31\|$ | Cloud-edge intelligence | Service response time/Task <br> processing node | Reduce the average <br> response time for multi-task <br> parallel AIGC services | Genetic algorithm |
| [186 | Cost-driven offloading for <br> DNN-Based applications | System costs/Number of <br> layers | Minimize costs of AIGC <br> services in a <br> cloud-edge-end <br> collaborative environment | Genetic algorithm based on <br> particle swarm optimization |
| [187 | Industrial edge intelligence | A weighted sum of task <br> execution time and energy <br> consumption/Task <br> assignment | Multi-objective <br> optimization of large-scale <br> AIGC tasks with multiple <br> connected devices | Generative coding <br> evolutionary algorithm |
| [188 | Computation offloading for <br> ML web apps | Inference time/Pre-sending <br> decisions | Reduce execution <br> overheads of AIGC tasks <br> with pre-sending snapshots | Hill climbing algorithm |
| [189 | Cooperative edge <br> intelligence | Quality of <br> experience/Offloading <br> decisions | Enhance vertical-horizontal <br> cooperation in multi-user <br> AIGC co-inference <br> scenarios | Federated multi-agent <br> reinforcement learning |

When implementing cloud-edge collaborative training and fine-tuning for generative AI models [192], it is important to consider specific algorithms or techniques that enable effective collaboration between cloud and edge servers [170], [193]. For example, FL and distributed training approaches can facilitate the collaboration process by allowing edge servers to train models locally and then send the updated weights to the cloud server for aggregation [194]. The division of responsibilities between cloud and edge servers can also greatly affect the overall efficiency and performance of the generative AI models. Therefore, it is crucial to discuss and implement appropriate schemes for determining which tasks are offloaded to the edge servers and which are performed on the cloud server. To provide AIGC services in edge intelligence-empowered IoT, offloading ML tasks to edge servers for remote execution is a promising approach for computation-intensive AI model inference [195]. For instance, in Fig 18, multiple lightweight ML models can be loaded into IoT devices, while largescale ML models can be installed and executed on edge servers [29]. Heterogeneous generative AI models can be deployed on mobile devices and edge servers according to their resource demands and service requirements [196]. However, the multiple attributes of ML tasks, such as accuracy, inference latency, and reliability, render the offloading problem of AIGC highly complex. Therefore, the authors in [184] propose an ML task offloading scheme to minimize task execution latency while guaranteeing inference accuracy. Considering error inference leading to extra delays in task processing, they initially model the inference process as M/M/1 queues, which are also applicable to the AIGC service process. Furthermore, the optimization problem of ML task execution is formulated as a Mixed-Integer Nonlinear Programming (MINLP) to minimize provisioning delay, which can be adopted in the inference process of AIGC services. To extend the deterministic environment in [184] into a more general environment, the authors in [185] first propose an adaptive translation mechanism to automatically and dynamically offload intelligent IoT applications. Then, they make predictive offloading decisions using a random forest regression model. Their experiments demonstrate that the proposed framework reduces response times for complex applications by half. Such ML methods can also be used to analyze AIGC network traffic to improve service delivery efficiency and reliability.

The success of edge-mobile collaboration for AIGC services is dependent on several factors, including the type of service, user characteristics, computational resources, and network conditions [4], [197], [198]. For instance, a real-time AIGC service may have different latency requirements compared to an offline service. Similarly, the required computational resources may vary depending on the model's complexity [199]. Additionally, the user profile, including location and device type, may affect the selection of edge servers for task offloading. Furthermore, network conditions such as bandwidth and packet loss rate can impact the reliability and latency of the service. Therefore, it is necessary to implement effective resource allocation and task offloading schemes to ensure highquality and low-latency AIGC services in dynamic and diverse environments. Cloud-edge collaborative intelligence enables local tasks to be offloaded to edge and cloud servers. AIGC can benefit from cloud-edge intelligence, as edge servers
can provide low-latency AIGC services while cloud servers can offer high-quality AIGC services. The authors in [32] develop a scheme called Neurosurgeon to select the optimal partitioning point based on model architectures, hardware platforms, network conditions, and load information at the servers to automatically partition the computation of tensors of DNNs between cloud and edge servers. Furthermore, the authors in [200] find that the layered approach can reduce the number of messages transmitted between devices by up to $97 \%$ while only decreasing the accuracy of models by a mere $3 \%$. However, multiple AIGC services should be considered in cloud-edge collaborative intelligence that differs in types (e.g., text, images, and videos) and their diverse quality of service (QoS) requirements [201]. In multi-task parallel scheduling [31], the genetic algorithm can also be used to make real-time model partitioning decisions. The authors in [186] propose a cost-driven strategy for AI application offloading through a self-adaptive genetic algorithm based on particle swarm optimization.

In industrial edge intelligence, where edge intelligence is embedded in the industrial IoT [187], [202]-[204], offloading computation tasks to edge servers is an efficient solution for self-organizing, autonomous decision-making, and rapid response throughout the manufacturing lifecycle, which is similarly required by mobile AIGC networks. Therefore, efficiently solving task assignment problems is crucial for effective generative AI model inference. However, the coexistence of multiple tasks among devices makes system response slow for various tasks. For example, text-based and image-based AIGC may coexist on the same edge device. As one solution, in [187], the authors propose a coding group evolution algorithm to solve large-scale task assignment problems, where tasks span the entire lifecycle of various products, including real-time monitoring, complex control, product structure computation, multidisciplinary cooperation optimization, and production process computation. Likewise, the AIGC lifecycle includes data collection, labeling, model training and optimization, and inference. Furthermore, a simple grouping strategy is introduced to parallel partition the solution space and accelerate the evolutionary optimization process. In contrast to VM-level adaptation to specific edge servers [205], the authors propose application-level adaptation for generic servers. The lighter adaptation framework in [188] further improves transmission time and user data privacy performance, including offloading and data/code recovery to generic edge servers.

Ensuring dependable task offloading is crucial in providing superior AIGC services with minimal latency in edge computing. For instance, data transmission redundancy can enhance dependability by transmitting data via multiple pathways to mitigate network congestion or failures. By incorporating these techniques, task offloading dependability in edge computing can be enhanced, thereby leading to more efficient and effective AIGC services. Most intelligent computing offloading solutions converge slowly, consume significant resources, and raise user privacy concerns [206], [207]. The situation is similar when leveraging learning-based approaches to make AIGC service offloading decisions. Consequently, the authors enhance multi-user QoE [208] for cooperative edge intel-

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-27.jpg?height=586&width=876&top_left_y=192&top_left_x=1080)

Fig. 19: An overview of edge caching in mobile AIGC networks. By caching the generative AI model on the edge servers, the latency of AIGC services can be reduced and the network congestion in the core network can be reduced.

ligence in [189] with federated multi-agent reinforcement learning. They formulate the cooperative offloading problem as a Markov Decision Process (MDP). The state is composed of current tasks, local loads, and edge loads. Learning agents select task processing positions to maximize multiuser QoE, which simultaneously considers service latency, energy consumption, task drop rate, and privacy protection. Similarly, AIGC service provisioning systems can easily adopt the proposed solution for maximizing QoE in AIGC services.

## C. Edge Caching

Edge caching is the delivery of low-latency content and computing services using the storage capacity of edge base stations and mobile devices [214], [215]. As illustrated in Fig. 19. in mobile AIGC networks, users can request AIGC services without accessing cloud data centers by caching generative $\mathrm{AI}$ models in edge servers and mobile devices. Unlike the cache in traditional content distribution networks, the generative $\mathrm{AI}$ model cache also requires computing resources to support its execution. Additionally, the generative AI model needs to gather user historical requests and profiles in context to provide personalized services during the AIGC service process. As shown in Table VI here are several KPIs for edge caching in AIGC networks:

- Model access delay: Model access latency is an important indicator of AIGC service quality. The latency is lowest when the generative AI model is cached in the mobile device [216]. The model access latency must also be calculated considering the delay in the wireless communication network when the edge server provides the generative AI model. Finally, the core network latency must be considered when the cloud provides the AIGC service.
- Backhaul traffic load: The load on the backhaul traffic is significantly reduced, as the requests and results of AIGC services do not need to go through the core network when

TABLE VI: Summary of scenarios, problems, performance metrics, and mathematical tools for edge caching in AIGC networks.

| Ref. | Scenarios | Performance <br> Metrics/Decision <br> Variables | Benefits/Challenges | Mathematical Tools |
| :---: | :---: | :---: | :---: | :---: |
| [209 | DL Model caching at the <br> edge | Runtime memory <br> consumption and loading <br> time/Model preload policy | Manage and utilize GPU <br> memories of edge servers <br> for caching generative AI <br> models | Cache replacement <br> algorithms |
| 210 | Caching many models at <br> the edge | Model load and execution <br> latency and monetary cost <br> /Caching eviction policy | Improve scalability of <br> mobile AIGC networks via <br> model-level caching <br> deployment and <br> replacement | Model utility calculation |
| [211 | Cache for mobile deep <br> vision applications | Latency, accuracy loss, <br> energy saving/Caching <br> policy, user selection, <br> transmit power, bandwidth <br> ratio | Caching for users' requests <br> for multimodal AIGC <br> services | Greedy algorithm |
| $[212$ | Cache for functions in <br> serverless computing | Execution time, cold start <br> proportion/Function <br> keep-alive policy | Keep generative AI models <br> alive and warm for <br> in-contextual inference | Greedy-dual based <br> approach |
| [213 | Knowledge caching for FL | Transmission latency and <br> energy <br> consumption/Caching <br> policy, user selection, <br> transmit power, bandwidth <br> ratio | Privacy-preserving model <br> caching via knowledge of <br> AIGC requests | Optimization theory |

the generative AI model is cached in the mobile edge network.

- Model hit rate: Similar to content hit rate, the model hit rate is an important metric for generative AI models in the edge cache. It can be used for future model exits and loading during model replacement.

As there is sufficient infrastructure and resources in the cloud computing infrastructure, the generative AI model can be fully loaded into the GPU memory for real-time service requests. In contrast, the proposed EdgeServe in [209] keeps models in main memory or GPU memory so that they can be effectively managed or used at the edge. Similar to traditional CDNs, the authors use model execution caches at edge servers to provide immediate AI delivery. In detail, there are mainly three challenges in generative AI model caching:

- Resource-constraint edge servers: Compared to the resource-rich cloud, the resources of servers in the edge network, such as GPU memory, are limited [217]. Therefore, caching all generative AI models on one edge server is infeasible.
- Model-missing cost: When the mobile device user requests AIGC, the corresponding model is missed if the generative AI model used to generate the AIGC is not cached in the current edge server [210]. In contrast to the instantly available AIGC service, if the generative AI model is missing, the edge server needs to send a model request to the cloud server and download the model, which causes additional overhead in terms of bandwidth and latency.
- Functionally equivalent models: The number of generative AI models is large and increases depending on the number of detailed tasks [218]. Meanwhile, AI models have similar functions in different applications, i.e., functionally equivalent. For example, for image recognition tasks, a large number of models with different architectures are proposed to recognize features in images, which have different model architectures and computation requirements.

To address these challenges, the authors in [209] formulate the problem of edge modeling as determining which DL models should be preloaded into memory and which should be discarded when the memory is full while satisfying the requirements of inferential response times. Fortunately, this edge model caching problem can be solved using existing cache replacement policies for edge content caching. The accuracies and computation complexities of DL models make this optimization problem more complicated than conventional edge caching problems. Similarly, for resource-constrained edge servers, the generative AI model can be dynamically deployed and replaced. However, an effective caching algorithm for loading and unloading the generative AI models to maximize the hit rate has not yet been investigated.

As the capabilities of AI services continue to grow and diversify, multiple models need to be deployed simultaneously at the edge to achieve various tasks, including classification, recognition, text/image/video generation [219]. Especially in mobile AIGC networks, multiple base models need to work together to generate a large amount of multimodal synthetic data. Many models play a synergistic role in the AIGC services at the edge of the network, while the support of multiple models also poses a challenge to the limited GPU memory of the edge servers. Therefore, the authors in [210] propose a model-level caching system with an eviction policy according to model characteristics and workloads. The model eviction policy is based on model utility calculation from cache miss
penalty and the number of requests. This model-aware caching approach introduces a new direction for providing AIGC services at mobile edge networks with heterogeneous requests. Experimental results show that compared to the non-penaltyaware eviction policy, the model load delay can be reduced by $1 / 3$. This eviction policy can also be adopted in the problem of which unpopular generative AI models should be unloaded.

At mobile AIGC networks, not only the generative AI model needs to be cached, but also the AIGC requests and results can be cached to reduce the latency of service requests in AIGC networks. To this end, the authors devise a principled cache design to accelerate the execution of CNN models by exploiting the temporal locality of video for continuous vision tasks to support mobile vision applications [220]. The authors in [211] propose a principled cache scheme, named DeepCache, to retrieve reusable results and reuse them within a fine-grained CNN by exploiting the temporal locality of the mobile video stream. In DeepCache, mobile devices do not need to offload any data to the cloud and can support the most popular models. Additionally, without requiring developers to retrain models or tune parameters, DeepCache caches inference results for unmodified CNN models. Overall, DeepCache can reduce energy consumption by caching content to reduce model inference latency while sacrificing a small fraction of model accuracy.

In serverless computing for edge intelligence, mobile devices can call functions of AIGC services at edge servers, which is more resource-efficient compared to container and virtual machine (VM)-based AIGC services. Nevertheless, such functions suffer from the cold-start problem of initializing their code and data dependencies at edge servers. Although the execution time of each function is usually short, initialization, i.e., fetching and installing prerequisite libraries and dependencies before execution, is time-consuming [221]. Fortunately, the authors in [212] show that the caching-based keep-alive policy can be used to address the cold-start problem by demonstrating that the keep-alive function is equivalent to caching. Finally, to balance the trade-off between server memory utilization and cold-start overhead, a greedy dualbased caching algorithm is proposed.

Frequently, a large-scale generative AI model can be partitioned into multiple computing functions that can be efficiently managed and accessed during training, fine-tuning, and inference. FL models can be cached on edge servers to facilitate user access to instances and updates, thus addressing user privacy concerns [222], [223]. For example, the authors in [213] propose a knowledge cache scheme for FL in which participants can simultaneously minimize training delay and training loss according to their preference. Their insight is that there are two stimulations for caching knowledge for FL [224]: i) training data sufficiency and ii) connectivity stability. Experimental results show that the proposed preference-driven caching policy, based on the preferences (i.e., demands or desires for global models) of participants in FL, can outperform the random policy when user preferences are intense. Therefore, preference-based generative AI model caching should be extensively investigated for providing personalized and customized AIGC services at edge servers.

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-29.jpg?height=580&width=862&top_left_y=195&top_left_x=1076)

Fig. 20: An overview of mobility management in mobile AIGC networks. The coverage of the mobile AIGC network will be significantly enhanced by UAV processing the user's server request and providing AIGC services.

## D. Mobility Management

Mobile edge intelligence for the Internet of Vehicles and Unmanned Aerial Vehicle (UAV) networks relies on effective mobility management solutions [232]-[235] to provide mobile AIGC services. Furthermore, UAV-based AIGC service distribution offers advantages such as ease of deployment, flexibility, and extensive coverage for enhanced edge intelligence [236], [237]. Specifically, UAVs, with their line-ofsight communication links, can extend the reach of edge intelligence [238]. For example, flexible UAVs equipped with AIGC servers enable users to access AIGC services with ultralow latency and high reliability, especially when fixed-edge servers are often overloaded in hotspot areas or expensive to deploy in remote areas, as illustrated in Fig. 20. In addition, UAV-enabled edge intelligence can be utilized to implement mobile AIGC content and service delivery.

As summarized in Table VII here are several KPIs for mobility management in AIGC networks:

- Task accomplishment ratio: The provisioning of AIGC services at mobile edge networks must consider the dynamic nature of users [239]. As a result, services must be completed before users leave the base station. To measure the effectiveness of mobility management in AIGC networks, the task completion rate can be used.
- Coverage enhancement: Vehicles and UAVs can serve as reconfigurable base stations to enhance the coverage of mobile AIGC networks [240], providing generative AI models and content to users anywhere and anytime.

In vehicular networks, intelligent applications, such as AIGC-empowered navigation systems, are reshaping existing transportation systems. In [225], the authors propose a joint vehicle-edge inference framework to optimize energy consumption while reducing the execution latency of DNNs. In detail, vehicles and edge servers determine an optimal partition point for DNNs and dynamically allocate resources for DNN execution. They propose a chemical reaction optimizationbased algorithm to accelerate convergence when solving the

TABLE VII: Summary of scenarios, problems, benefits/challenges, and mathematical tools for mobility management.

| Ref. | Scenarios | Performance <br> Metrics/Problems | Benefits/Challenges | Mathematical Tools |
| :---: | :---: | :---: | :---: | :---: |
| \|225 | Jointing vehicle-edge deep <br> neural network inference | Latency, failure rate/CPU <br> frequency | Robust AIGC service <br> provisioning via layer-level <br> offloading | Chemical reaction <br> optimization |
| [226 | Vehicular edge intelligence | Weighted average <br> completion time and task <br> acceptance ratio/Task <br> dispatching policy | Provisioning AIGC service <br> in multi-vehicle <br> environments with motion <br> prediction | Greedy algorithm |
| \|227| | Mobility-enhanced edge <br> intelligence | Task completion ratio and <br> model accuracy/Offloading <br> redundancy, task <br> assignment, beam selection | Sustainable AIGC service <br> provisioning with mobility <br> management | FL |
| \|228| | Edge intelligence-assisted <br> IoV | Average delay and energy <br> consumption/Transmission <br> decision, task offloading <br> decision, bandwidth, and <br> computation resource <br> allocation | Flexible network model <br> selection for AIGC services <br> for balancing the tradeoff <br> adaptively | Quantum-inspired <br> reinforcement learning |
| [229 | Cooperative edge <br> intelligence in IoV | Average delay and energy <br> consumption/Trajectory <br> prediction accuracy | Optimize AIGC service <br> with spatial and temporal <br> correlations of users' <br> requests | Hybrid stacked autoencoder <br> learning |
| [230 | UAVs as an intelligent <br> service | Model accuracy and energy <br> consumption/Number of <br> local iterations | Provision AIGC services <br> via a network of UAVs | Greedy algorithm |
| \|231 | Knowledge <br> distillation-empowered <br> edge intelligence | Accuracy and inference <br> delay/Saze of model <br> parameters | Visual information-aided <br> generative AI model <br> deployment and inference <br> scheduling | Knowledge distillation |

resource allocation problem. This framework offers insights for implementing mobile AIGC networks, where vehicles can collaborate with base stations to provide real-time AIGC services based on DNNs during their movement.

AIGC applications require sufficient processing and memory resources to perform extensive AIGC services [241][244]. However, resource-constrained vehicles cannot meet the QoS requirements of the tasks. The authors in [226] propose a distributed scheduling framework that develops a prioritydriven transmission scheduling policy to address the dynamic network topologies of vehicle networks and promote vehicle edge intelligence. To meet the various QoS requirements of intelligent tasks, large-volume tasks can be partitioned and sequentially uploaded. Additionally, the impact of vehicle motion on task completion time and edge server load balancing can be independently handled by intelligent task processing requests. The effectiveness of the proposed framework is demonstrated in single-vehicle and multi-vehicle environments through simulation and deployment experiments. To facilitate smart and green vehicle networks [227], the real-time accuracy of AI tasks, such as generative AI model inference, can be monitored through on-demand model training using infrastructure vehicles and opportunity vehicles.

The heterogeneous communication and computation requirements of AIGC services in highly dynamic, time-varying Internet of Vehicles (IoV) warrant further investigation [245][248]. To dynamically make transmission and offload decisions, the authors in [228] formulate a Markov decision process for time-varying environments in their joint commu- nication and computation resource allocation strategy. Finally, they develop a quantum-inspired reinforcement learning algorithm, in which quantum mechanisms can enhance learning convergence and performance. The authors in [229] propose a stacked autoencoder to capture spatial and temporal correlations to combine road traffic management and data network traffic management. To reduce vehicle energy consumption and learning delay, the proposed learning model can minimize the required signal traffic and prediction errors. Consequently, the accuracy of AIGC services based on autoencoder techniques can be improved through this management framework.

With UAV-enhanced edge intelligence, UAVs can serve as aerial wireless base stations, edge computing servers, and edge caching providers in mobile AIGC networks [249], [250]. To demonstrate the performance of UAV-enhanced edge intelligence while preserving user privacy at mobile edge networks, the authors in [230] use UAV-enabled FL as a use case. Moreover, the authors suggest that flexible switching between compute and cache services using adaptive scheduling UAVs is a topic for future research. Therefore, flexible AIGC service provisioning and UAV-based AIGC delivery are essential for satisfying real-time service requirements and reliable generation. In this regard, the authors in [231] propose a visually assisted positioning solution for UAV-based AIGC delivery services where GPS signals are weak or unstable. Specifically, knowledge distillation is leveraged to accelerate inference speed and reduce resource consumption while ensuring satisfactory model accuracy.

## E. Incentive Mechanism

As suitable incentive mechanisms are designed, more edge nodes participate in and contribute to the AIGC services [146], [255]-[257]. This increases the computational capacity of the system. In addition, the nodes are motivated to earn rewards by providing high-quality services. Thus, the overall quality of AIGC services is improved. Finally, nodes are encouraged to engage in secure operations without security concerns by recording resource transactions through the blockchain.

As listed in Table VIII, here are several KPIs for incentive mechanisms in AIGC networks:

- Social welfare: AIGC's social welfare is the sum of the value of AIGC's services to the participants of the current network. Higher social welfare means that more AIGC users and AIGC service providers are participating in the AIGC network and providing high-value AIGC services within the network.
- Revenue: Providers of AIGC use a large amount of computing and energy resources to provide AIGC, which may be offset by revenue from AIGC users. The higher the revenue, the more the AIGC service provider can be motivated to improve the AIGC service to a higher quality.
- Economic properties: In AIGC networks, AIGC providers and users should be risk-neutral, which indicates the incentive mechanisms should satisfy economic properties, e.g., individually rational, incentive compatible, and budget balance [258].

While edge learning has several promising benefits, the learning time for satisfactory performance and appropriate monetary incentives for resource providers are nontrivial challenges for AIGC. In [251], [259], [260], where mobile devices are connected to the edge server, the authors design the incentive mechanism for efficient edge learning. Specifically, mobile devices collect data and train private models locally with computational resources based on the price of edge servers in each training round. Then, the updated models are uploaded to the edge server and aggregated to minimize the global loss function. Furthermore, the authors in [261] not only analyze the optimal pricing strategy but also use Deep Reinforcement Learning to learn the pricing strategy to obtain the optimal solution in each round in a dynamic environment and with incomplete information. In the absence of prior knowledge, the DRL agent can learn from experience to find the optimal pricing strategy that balances payment and training time. To extend [251] to long-term incentive provisioning, the authors in [252] propose a long-term incentive mechanism for edge learning frameworks. To obtain the optimal shortterm and long-term pricing strategies, the hierarchical deep reinforcement learning algorithm is used in the framework to improve the model accuracy with budget constraints.

In the process of fine-tuning the AIGC edge, the incentives described above can be used to balance the time and adaptability of the fine-tuned generative AI model. In providing incentives to AIGC service providers, the quality of AIGC services also needs to be considered in the incentive mechanism. The authors in [253] propose a quality-aware
FL framework to prevent inferior model updates from degrading the global model quality. Specifically, based on an AI model trained from historical learning results, the authors estimate the learning quality of mobile devices. To motivate participants to contribute high-quality services, the authors propose a reverse auction-based incentive mechanism under the recruitment budget of edge servers, taking into account the model quality. Finally, the authors propose an algorithm for integrating the model quality into the aggregation process and for filtering non-optimal model updates to further optimize the global learning model.

Traditionally, resource utilization is inefficient, and trading mechanisms are unfair in cloud-edge computing power trading [262] for AIGC services. To address this issue, the authors in [254] develop a general trading framework for computing power grids. As illustrated in Fig. 22, the authors solve the problem of the under-utilization of computing power with AI consumers in this framework. The computing-power trading problem is first formulated as a Stackelberg game and then solved with a profit-driven multi-agent reinforcement learning algorithm. Finally, a blockchain is designed for transaction security in the trading framework. In mobile AIGC networks with multiple AIGC service providers and multiple AIGC users, the Stackelberg game and its extension can still provide a valid framework for equilibrium analysis. In addition, multi-agent reinforcement learning also learns the equilibrium solution of the game by exploration and exploitation in the presence of incomplete information about the game.

## F. Security and Privacy

Mobile AIGC networks leverage a collaborative computing framework on the cloud side to provide AIGC services, utilizing a large amount of heterogeneous data and computing power [263]-[266]. When mobile users are kind, AIGC can greatly enhance their creativity and efficiency. However, malicious users can also utilize AIGC for destructive purposes, posing a threat to users in mobile edge networks. For example, AI-generated text can be used by malicious users to complete phishing emails, thus compromising the security and privacy of normal users [11]. To ensure secure AIGC services, providers must choose trusted AIGC solutions and securely train AI models while providing secure hints and answers to AIGC service users [267].

1) Privacy-preserving AIGC Service Provisioning: During the lifecycle of providing AIGC services, privacy information in large-scale datasets and user requests needs to be kept secure to prevent privacy breaches. In mobile AIGC networks, the generation and storage of data for generative AI model training occur at edge servers and mobile devices [268][270]. Unlike resourceful cloud data centers, edge and mobile layers have limited defense capacities against various attacks. Fortunately, several privacy-preserving distributed learning frameworks, such as FL [271], [272], have been proposed to empower privacy-preserving generative AI model fine-tuning and inference at mobile AIGC networks. In preserving user privacy in AIGC networks, FL is a distributed ML approach that allows users to transmit local models instead of data

TABLE VIII: Summary of scenarios, problems, benefits/challenges, and mathematical tools of incentive mechanism.

| Ref. | Scenarios | Problems | Benefits/Challenges | Mathematical Tools |
| :---: | :---: | :---: | :---: | :---: |
| 251. | Efficient edge learning | A weighted sum of training <br> time and payment/Total <br> payment and training time | Incentivize AIGC service <br> providers with <br> heterogeneous resources <br> under the uncertainty of <br> edge network bandwidth | Deep reinforcement <br> learning |
| 252 | Efficient edge learning | Model accuracy, number of <br> training rounds, time <br> efficiency/The total price | Long-term incentive <br> mechanism for AIGC <br> services with long-term and <br> short-term pricing strategies | Hierarchical deep <br> reinforcement learning |
| 253 | Model accuracy and loss <br> reduction/Learning quality <br> estimation and <br> quality-aware incentive <br> mechanism | Estimate the performance <br> of AIGC services with <br> privacy-preserving methods <br> for distributing proper <br> incentives | Reverse auction |  |

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-32.jpg?height=588&width=878&top_left_y=947&top_left_x=168)

Fig. 21: Federated Learning in mobile AIGC networks, including the local model training at mobile devices, global aggregation at edge servers, and cross-server model trading.

during model training [204], [273], [274]. Specifically, as illustrated in Fig. 21, there are two major approaches to employing FL in AIGC networks

- Secure aggregation: While FL is being learned, the mobile devices send local updates to edge servers for global aggregation. During global aggregation, authenticated encryption allows the use of secret sharing mechanisms.
- Differential privacy: Differential privacy can prevent FL servers from identifying the owners of a local update. Differential privacy is similar to secure aggregation in that it prevents FL servers from identifying owners of local updates.

Therefore, in [275], the authors propose a differential private federated generative model to synthesize representative examples of private data. With guaranteed privacy, the proposed model can solve many common data problems without human intervention. Moreover, in [276], the authors propose an FLbased generative learning scheme to improve the efficiency and robustness of GAN models. The proposed scheme is particularly effective in the presence of varying parallelism and highly skewed data distributions. To find an inherent cluster structure in users' data and unlabeled datasets, the authors propose in [277] the unsupervised Iterative Federated Clustering algorithm, which uses generative models to deal with the statistical heterogeneity that may exist among the participants of FL. Since the centralized FL frameworks in [276], [277] might raise security concerns and risk single-point failure, the authors propose in [278] a decentralized FL framework based on a ring topology and deeply generated models. On the one hand, a method for synchronizing the ring topology can improve the communication efficiency and reliability of the system. On the other hand, generative models can solve data-related problems, such as incompleteness, low quality, insufficient quantity, and sensitivity. Finally, an InterPlanetary File System (IPFS)-based data-sharing system is developed to reduce data transmission costs and traffic congestion.

2) Secure AIGC Service Provisioning: Given the numerous benefits of provisioning AIGC services in mobile and edge layers, multi-tier collaboration among cloud servers, edge servers, and mobile devices enables ubiquitous AIGC service provision by heterogeneous stakeholders [151], [279]-[281]. A trustworthy collaborative AIGC service provisioning framework must be established to provide reliable and secure AIGC services. Compared to central cloud AIGC providers, mobile and edge AIGC providers can customize AIGC services by collaborating with many user nodes while distributing data to different devices [282]. Therefore, a secure access control mechanism is required for multi-party content streaming to ensure privacy and security. However, the security of AIGC transmission cannot be ensured due to various attacks on mobile AIGC networks [283]. Fortunately, blockchain [283][286], based on distributed ledger technologies, can be utilized to explore a secure and reliable AIGC service provisioning framework and record resource and service transactions to encourage data sharing among nodes, forming a trustworthy and active mobile AIGC ecosystem [287]. As illustrated in Fig. 22, there are several benefits that blockchain brings to mobile AIGC networks [26]:

![](https://cdn.mathpix.com/cropped/2024_06_04_05b0d12c99fc32622f90g-33.jpg?height=610&width=899&top_left_y=186&top_left_x=147)

Fig. 22: Blockchain in mobile AIGC networks 254,, including the AIGC application layer, blockchain layer, and computingpower network layers, for provisioning AIGC services.

- Computing and Communication Management: Blockchain enables heterogeneous computing and communication resources to be managed securely, adaptively, and efficiently in mobile AIGC networks [288].
- Data Administration: By recording AIGC resource and service transactions in blockchain with smart contracts, data administration in mobile AIGC networks is made profitable, collaborative, and credible.
- Optimization: During optimization in AIGC services, the blockchain always provides available, complete, and secure historical data for input to optimization algorithms.

For instance, the authors in [289] propose an edge intelligence framework based on deep generative models and blockchain. To overcome the accuracy issue of the limited dataset, GAN is leveraged in the framework to synthesize training samples. Then, the output of this framework is confirmed and incentivized by smart contracts based on the proof-of-work consensus algorithm. Furthermore, the multimodal outputs of AIGC can be minted as NFTs and then recorded on the blockchain. The authors in [290] develop a conditional generative model to synthesize new digital asset collections based on the historical transaction results of previous collections. First, the context information of NFT collections is extracted based on unsupervised learning. Based on the historical context, the newly minted collections are generated based on future token transactions. The proposed generative model can synthesize new NFT collections based on the contexts, i.e., the extracted features of previous transactions.

## G. Lessons Learned

1) Multi-Objective Quality of AIGC Services: In mobile AIGC networks, the quality of AIGC services is determined by several factors, including model accuracy, service latency, energy consumption, and revenue. Consequently, AIGC service providers must optimally allocate edge resources to satisfy users' multidimensional quality requirements for AIGC ser- vices [176]. Moreover, the migration of AIGC tasks and computations can enhance the reliability and efficiency of AIGC services. Notably, dynamic network conditions in the edge network necessitate users to make online decisions to achieve load balancing and efficient use of computing resources. A variety of methodologies are proposed, enhancing the multiobjective quality of AIGC services within mobile edge networks [153]. The techniques encompass multi-objective optimization among QoS, QoE, latency, and resource consumption. The primary objective of designing these strategies is to optimize key parameters such as accuracy, latency, resource consumption, and user satisfaction. The benefits including heightened performance and superior user experience, are attained, albeit at the potential cost of an increase in complexity, resource consumption, and potential privacy issues. Attaining high-quality AIGC services requires proper considerations and practices to address the challenges discussed above, meet the quality requirements of multiple objectives, and improve user satisfaction and service quality.
2) Edge Caching for Efficient Delivery of AIGC Services: Edge caching plays a pivotal role in the efficient delivery of AIGC services in mobile AIGC networks. Tackling the challenges of constrained-memory edge servers, modelmissing costs, and functionally equivalent models is essential for optimizing caching policies. Developing model-aware caching approaches, investigating preference-driven caching policies, and implementing principled cache designs to reduce latency and energy consumption are promising directions for enhancing the performance of mobile AIGC networks. In the quest for the efficient delivery of AIGC services via edge caching in mobile edge networks, the need for well-designed edge caching algorithms is emphasized [216]. The benefits associated with these algorithms include enhanced efficiency, decreased latency, and improved dependability. Conversely, the challenges that may arise from these strategies include escalated complexity, heightened costs, and potential privacy concerns. As AI services continue to evolve, further research in caching strategies is crucial for providing effective, personalized, and low-latency AIGC services for mobile users.
3) Preference-aware AIGC Service Provisioning: Offering AIGC services based on user preferences not only improves user satisfaction but also reduces service latency and resource consumption in mobile edge networks. To implement preference-based AIGC service delivery, AIGC service providers must first collect historical user data and analyze it thoroughly. In providing AIGC services, the service provider makes personalized recommendations and adjusts its strategy according to user feedback. The exploration of preferenceaware AIGC service provisioning is conducted considering several techniques, which include collaborative filtering, DRL, context awareness, user profiling, and multi-objective optimization. Although user preferences play a significant role in AIGC service provision, it is essential to use and manage this information properly to protect user privacy.
4) Life-cycle Incentive Mechanism throughout AIGC Services: In mobile AIGC networks, the entire life cycle of AIGC services necessitates appropriate incentives for participants. A single AIGC service provider cannot provide AIGC ser-
vices alone. Throughout the data collection, pre-training, finetuning, and inference of AIGC services, stakeholders with heterogeneous resources require reasonable incentives and must share the benefits according to their contributions. Conversely, from the users' perspective, evaluation mechanisms must be introduced. For instance, users can assess the reputation of AIGC service providers based on their transaction history to promote service optimization and improvement. Ultimately, the provisioning and transmission logs of AIGC services can also be recorded in a tamper-proof distributed ledger. Incentive strategies for participants in the life cycle of AIGC services in mobile edge networks are also examined. The use of smart contracts, distributed ledger technology, evaluation mechanisms, and incentive design is proposed as a means to strengthen collaboration and enhance the overall quality of AIGC services [254]. These methodologies introduce automation, transparency, and improved reputation, which are seen as distinct advantages.
5) Blockchain-based System Management of Mobile AIGC Networks: Furthermore, mobile AIGC networks connect heterogeneous user devices to edge servers and cloud data centers. This uncontrolled demand for content generation introduces uncertainty and security risks into the system. Therefore, secure management and auditing methods are required to manage devices in edge environments, such as dynamically accessing, departing, and identifying IoT devices. In the traditional centralized management architecture, the risk of central node failure is unavoidable. Thus, a secure and reliable monitoring and equipment auditing system should be developed. Lastly, we analyze a suite of techniques aimed at improving blockchain-based system management of mobile AIGC networks. Such techniques include blockchain-based data administration, secure management and auditing methods, collaborative infrastructure, decentralized management architecture, and blockchain-based optimization [146].

## VII. Future ReSEARCH DireCTIONS AND OpEn ISSUES

In this section, we discuss future research directions and open issues from the perspectives of networking and computing, ML, and practical implementation.

## A. Networking and Computing Issues

1) Decentralized Mobile AIGC Networks: With the advancement of blockchain technologies [291], decentralized mobile AIGC networks can be realized based on distributed data storage, the convergence of computing and networking, and proof-of-ownership of data [287]. Such a decentralized network structure, enabled by digital identities and smart contracts, can protect AIGC users' privacy and data security. Furthermore, based on blockchain technologies, mobile AIGC networks can achieve decentralized management of the entire lifecycle of AIGC services. Therefore, future research should investigate specific consensus mechanisms [291], [292], offchain storage systems, and token structures for the deployment of decentralized mobile AIGC networks [145].
2) Sustainability in Mobile AIGC Networks: In mobile AIGC networks, the pre-training, fine-tuning, and inference of generative AI models typically consume a substantial amount of computing and networking resources [30], [293]. Hence, future research can focus on the green operations of mobile AIGC networks that provide AIGC services with minimal energy consumption and carbon emissions. To this end, effective algorithms and frameworks should be developed to operate mobile AIGC networks under dynamic service configurations, operating modes of edge nodes, and communication links. Moreover, intelligent resource management and scheduling techniques can also be proposed to balance the tradeoff between service quality and resource consumption [294].
3) Wireless Communications in Mobile AIGC Networks: The influence of wireless communications on AIGC services is a critical area for future research. A key aspect to investigate is the robustness of AIGC services to the challenges posed by wireless communications [143]. This includes understanding how factors such as transmit power, fading, and device mobility within an edge network can affect the performance of distributed diffusion model-based AIGC computing [225]. Initial research in this area, such as the study in [145], has shown that despite the increase in bit error probability, distributed AIGC computing exhibits relatively high robustness. Further exploration of this robustness, as well as the development of strategies to enhance it, could significantly improve the performance and reliability of AIGC services in wireless networks. This can involve, for example, the development of adaptive physical layer transmission strategies [295] that take into account the current state of the wireless channel or the design of error correction mechanisms that can recover from bit errors introduced during wireless transmission [296], [297]. In addition, the use of AI-generated optimization solutions, particularly diffusion models, to overcome the challenges posed by the wireless environment and generate optimal solutions for network design is a promising avenue for future research. This can involve the development of AI-generated incentive mechanisms to promote semantic information exchange among users, as demonstrated by the authors [143]. Such mechanisms can help to create an optimal contract that adheres to the utility threshold constraints of the semantic information provider while maximizing the utility of the semantic information recipient.

High-quality data resources are also critical for the sustainability of mobile AIGC networks [144|. The performance of generative models depends not only on effective network architectures but also on the quality of training datasets [298]. However, as AIGC becomes pervasive, training datasets are gradually replaced by synthesized data that might be irrelevant to real data. Therefore, improving the quality and reliability of data in mobile AIGC networks, such as through multimodal data fusion and incremental learning technology, can further enhance the accuracy and performance of the models.

## B. Machine Learning Issues

1) Generative AI Model Compression: As generative AI models become increasingly complex, model compression
techniques are becoming more important to reduce service latency and resource consumption in provisioning AIGC services [299]. Fortunately, several techniques have been developed for generative AI model compressions, such as pruning, quantization, and knowledge distillation. First, pruning involves removing unimportant weights from the model, while quantization reduces the precision of the weights [300]. Then, knowledge distillation involves training a smaller model to mimic the larger model's behavior. Future research on generative AI model compression might continue to focus on developing and refining these techniques to improve their efficiency and effectiveness for deploying generative AI models in edge nodes and mobile devices. It is necessary to consider the limited resources of such devices and develop specialized compression techniques that can balance model size and accuracy.
2) AI-generated Network Design: Generative AI models have various potential applications in mobile networks, including design, analysis, control, monitoring, and traffic prediction [1], [301]. They can be utilized to create efficient network architectures, understand network behavior, predict network loads, develop network control algorithms, detect anomalies, and predict future network traffic patterns and demands [1]. Future research directions in machine learning for mobile AIGC networks can focus on improving the efficiency and effectiveness of existing applications, exploring new applications and use cases, and addressing the challenges posed by the unique characteristics of mobile networks, such as mobility, limited resources, and privacy concerns.
3) Privacy-preserving AIGC Services: To provide privacypreserving AIGC services, it is necessary to consider privacy computing techniques in both generative AI model training and inference [19], [142]. Techniques such as differential privacy, secure multi-party computation, and homomorphic encryption can be used to protect sensitive data and prevent unauthorized access. Differential privacy involves adding noise to the data to protect individual privacy, while secure multi-party computation allows multiple parties to compute a function without revealing their inputs to one another. Homomorphic encryption enables computations to be performed on encrypted data without decryption. To successfully deploy generative AI models in edge nodes and mobile devices, the limited resources of such devices should be considered and specialized techniques that can balance privacy and performance should be developed [158]. Additionally, concerns such as data ownership and user privacy leakage should be taken into account.

## C. Practical Implementation Issues

1) Integrating AIGC and Digital Twins: Digital twins enable the maintenance of representations to monitor, analyze, and predict the status of physical entities [302]. On one hand, the integration of AIGC and digital twin technologies has the potential to significantly improve the performance of mobile AIGC networks. By creating virtual representations of physical mobile AIGC networks, service latency, and quality can be optimized through the analysis of historical data and online predictions. Furthermore, AIGC can also enhance digital twin applications by reducing the time required for designers to create simulation entities. However, several issues need to be considered during the integration of AIGC and DTs, such as efficient and secure synchronization.
2) Immersive Streaming: AIGC can create immersive streaming content, such as AR and VR, that can transport viewers to virtual worlds [303], which can be used in various applications such as education, entertainment, and social media. Immersive streaming can enhance the AIGC delivery process by providing a platform for viewers to interact with the generated content in real-time. However, combining AIGC and immersive streaming raises some concerns. Future research should focus on addressing the potential for biased content generation by the AIGC algorithms and the high bandwidth requirements of immersive streaming, which can cause latency issues, resulting in the degradation of the viewer's experience.
3) Alignment: In human-oriented applications that involve digital humans and avatars, the alignment of generative AI models [52], [304], [305] in mobile AIGC networks should be well-investigated for safety and ethnicity. There are several potential research directions for AI alignment, such as personalized AI alignment, ethical guidelines for AI-generated content, trust and transparency, emotional alignment, cultural alignment, and robustness to adversarial attacks. By focusing on these areas, future AI alignment research in mobile AIGC networks can help maintain a user-centric, respectful, and ethically responsible approach for mobile AIGC networks and their applications.

## VIII. CONCLUSIONS

In this paper, we have focused on the deployment of mobile AIGC networks, which serve generative AI models, services, and applications at mobile edge networks. We have discussed the background and fundamentals of generative models and the lifecycle of AIGC services at mobile AIGC networks. We have also explored AIGC-driven creative applications and use cases for mobile AIGC networks, as well as the implementation, security, and privacy challenges of deploying mobile AIGC networks. Finally, we have highlighted some future research directions and open issues for the full realization of mobile AIGC networks.

## REFERENCES

[1] H. Du, R. Zhang, Y. Liu, J. Wang, Y. Lin, Z. Li, D. Niyato, J. Kang, Z. Xiong, S. Cui et al., "Beyond deep reinforcement learning: A tutorial on generative diffusion models in network optimization," arXiv preprint arXiv:2308.05384, 2023.

[2] E. Cetinic and J. She, "Understanding and creating art with AI: Review and outlook," ACM Transactions on Multimedia Computing, Communications, and Applications, vol. 18, no. 2, pp. 1-22, Feb. 2022.

[3] L.-H. Lee, Z. Lin, R. Hu, Z. Gong, A. Kumar, T. Li, S. Li, and P. Hui, "When creators meet the metaverse: A survey on computational arts," arXiv preprint arXiv:2111.13486, Apr. 2021.

[4] W. Wu, C. Zhou, M. Li, H. Wu, H. Zhou, N. Zhang, X. S. Shen, and W. Zhuang, "AI-native network slicing for 6G networks," IEEE Wireless Communications, vol. 29, no. 1, pp. 96-103, Apr. 2022.

[5] Y. Wang, Y. Pan, M. Yan, Z. Su, and T. H. Luan, "A survey on ChatGPT: AI-generated contents, challenges, and solutions," arXiv preprint arXiv:2305.18339, Feb. 2023.

[6] S. Bond-Taylor, A. Leach, Y. Long, and C. G. Willcocks, "Deep generative modelling: A comparative review of VAEs, GANs, normalizing flows, energy-based and autoregressive models," IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 73277347, Sep. 2021.

[7] A. Radford, J. W. Kim, C. Hallacy, A. Ramesh, G. Goh, S. Agarwal, G. Sastry, A. Askell, P. Mishkin, J. Clark et al., "Learning transferable visual models from natural language supervision," in Proceedings of the International Conference on Machine Learning, Virtual Conference, Jul. 2021, pp. 8748-8763.

[8] A. Ramesh, M. Pavlov, G. Goh, S. Gray, C. Voss, A. Radford, M. Chen, and I. Sutskever, "Zero-shot text-to-image generation," in Proceedings of the International Conference on Machine Learning, Virtual Conference, Jul. 2021, pp. 8821-8831.

[9] A. Ramesh, P. Dhariwal, A. Nichol, C. Chu, and M. Chen, "Hierarchical text-conditional image generation with CLIP latents," arXiv preprint arXiv:2204.06125, Apr. 2022.

[10] S. Huang, P. Grady, and GPT-3, "GenerativeAI: A creative new world," "Accessed Feb. 4, 2023", [Online]. Available: https://www.sequoiacap. com/article/generative- $\{\mathrm{AI}\}$-a-creative-new-world/

[11] E. Crothers, N. Japkowicz, and H. Viktor, "Machine generated text: A comprehensive survey of threat models and detection methods," arXiv preprint arXiv:2210.07321, Oct. 2022.

[12] O. AI, "Chatgpt: Optimizing language models for dialogue," "Accessed Feb. 4, 2023", [Online]. Available: https://openai.com/blog/chatgpt/

[13] J. Ho, W. Chan, C. Saharia, J. Whang, R. Gao, A. Gritsenko, D. P. Kingma, B. Poole, M. Norouzi, D. J. Fleet et al., "Imagen video: High definition video generation with diffusion models," arXiv preprint arXiv:2210.02303, Oct. 2022.

[14] M. Kim, A. DeRieux, and W. Saad, "A bargaining game for personalized, energy efficient split learning over wireless networks," in 2023 IEEE Wireless Communications and Networking Conference (WCNC), Glasgow, United Kingdom, May 2023, pp. 1-6.

[15] X. Wang, Y. Han, V. C. Leung, D. Niyato, X. Yan, and X. Chen, "Convergence of edge computing and deep learning: A comprehensive survey," IEEE Communications Surveys \& Tutorials, vol. 22, no. 2, pp. 869-904, Jan. 2020.

[16] M. Westerlund, "The emergence of deepfake technology: A review," Technology Innovation Management Review, vol. 9, no. 11, pp. 40-53, Nov. 2019.

[17] X. Yuan, L. Pu, L. Jiao, X. Wang, M. Yang, and J. Xu, "When computing power network meets distributed machine learning: An efficient federated split learning framework," arXiv preprint arXiv:2305.12979, Mar. 2023.

[18] J. Zhang and K. B. Letaief, "Mobile edge intelligence and computing for the Internet of vehicles," Proceedings of the IEEE, vol. 108, no. 2, pp. 246-261, Jun. 2019.

[19] W. Y. B. Lim, N. C. Luong, D. T. Hoang, Y. Jiao, Y.-C. Liang, Q. Yang, D. Niyato, and C. Miao, "Federated learning in mobile edge networks: A comprehensive survey," IEEE Communications Surveys \& Tutorials, vol. 22, no. 3, pp. 2031-2063, Apr. 2020.

[20] M. Makhmutov, S. Varouqa, and J. A. Brow, "Survey on copyright laws about music generated by artificial intelligence," in Proceedings of the IEEE Symposium Series on Computational Intelligence, ACT, Australia, Jan. 2020, pp. 3003-3009.

[21] M. Chen, D. Gündüz, K. Huang, W. Saad, M. Bennis, A. V. Feljan, and H. V. Poor, "Distributed learning in wireless networks: Recent progress and future challenges," IEEE Journal on Selected Areas in Communications, vol. 39, no. 12, pp. 3579-3605, Oct. 2021.

[22] F. Zhan, Y. Yu, R. Wu, J. Zhang, and S. Lu, "Multimodal image synthesis and editing: A survey," arXiv preprint arXiv:2112.13592, Dec. 2021.

[23] X. Shen, J. Gao, W. Wu, M. Li, C. Zhou, and W. Zhuang, "Holistic network virtualization and pervasive network intelligence for 6G," IEEE Communications Surveys \& Tutorials, vol. 24, no. 1, pp. 1-30, Dec. 2021.

[24] K. B. Letaief, Y. Shi, J. Lu, and J. Lu, "Edge artificial intelligence for 6G: Vision, enabling technologies, and applications," IEEE Journal on Selected Areas in Communications, vol. 40, no. 1, pp. 5-36, Nov. 2021.

[25] H. Cao, C. Tan, Z. Gao, G. Chen, P.-A. Heng, and S. Z. Li, "A survey on generative diffusion model," arXiv preprint arXiv:2209.02646, Sep. 2022.

[26] X. Wang, X. Ren, C. Qiu, Z. Xiong, H. Yao, and V. C. Leung, "Integrating edge intelligence and blockchain: What, why, and how," IEEE Communications Surveys \& Tutorials, vol. 24, no. 4, pp. 21932229, Jul. 2022.
[27] M. Xu, W. C. Ng, W. Y. B. Lim, J. Kang, Z. Xiong, D. Niyato, Q. Yang, X. S. Shen, and C. Miao, "A full dive into realizing the edgeenabled metaverse: Visions, enabling technologies, and challenges," IEEE Communications Surveys \& Tutorials, vol. 25, no. 1, pp. 656700, Nov. 2023.

[28] S. Nyatsanga, T. Kucherenko, C. Ahuja, G. E. Henter, and M. Neff, "A comprehensive review of data-driven co-speech gesture generation," arXiv preprint arXiv:2301.05339, Jan. 2023.

[29] Z. Zhou, X. Chen, E. Li, L. Zeng, K. Luo, and J. Zhang, "Edge intelligence: Paving the last mile of artificial intelligence with edge computing," Proceedings of the IEEE, vol. 107, no. 8, pp. 1738-1762, Jun. 2019.

[30] Y. Mao, C. You, J. Zhang, K. Huang, and K. B. Letaief, "A survey on mobile edge computing: The communication perspective," IEEE Communications Surveys \& Tutorials, vol. 19, no. 4, pp. 2322-2358, Aug. 2017.

[31] Z. Chen, J. Hu, X. Chen, J. Hu, X. Zheng, and G. Min, "Computation offloading and task scheduling for DNN-based applications in cloudedge computing," IEEE Access, vol. 8, pp. 115 537-115 547, Jun. 2020.

[32] Y. Kang, J. Hauswald, C. Gao, A. Rovinski, T. Mudge, J. Mars, and L. Tang, "Neurosurgeon: Collaborative intelligence between the cloud and mobile edge," ACM SIGARCH Computer Architecture News, vol. 45, no. 1, pp. 615-629, Mar. 2017.

[33] H. Zhang and B. Di, "Intelligent omni-surfaces: Simultaneous refraction and reflection for full-dimensional wireless communications," IEEE Communications Surveys \& Tutorials, vol. 24, no. 4, pp. 19972028, Aug. 2022.

[34] D. Huang, P. Chen, R. Zeng, Q. Du, M. Tan, and C. GAN, "Locationaware graph convolutional networks for video question answering," in Proceedings of the AAAI Conference on Artificial Intelligence, vol. 34, no. 07, New York, New York, Feb. 2020, pp. 11021-11028.

[35] H. Zhang, B. Di, K. Bian, Z. Han, H. V. Poor, and L. Song, "Toward ubiquitous sensing and localization with reconfigurable intelligent surfaces," Proceedings of the IEEE, vol. 110, no. 9, pp. 1401-1422, May 2022.

[36] X. Wang, M. Chen, T. Taleb, A. Ksentini, and V. C. Leung, "Cache in the air: Exploiting content caching and delivery techniques for $5 \mathrm{G}$ systems," IEEE Communications Magazine, vol. 52, no. 2, pp. 131139, Feb. 2014.

[37] S. Huang, H. Zhang, X. Wang, M. Chen, J. Li, and V. C. Leung, "Fine-grained spatio-temporal distribution prediction of mobile content delivery in 5G ultra-dense networks," IEEE Transactions on Mobile Computing, pp. 1-14, Dec. 2022.

[38] Z. Q. Liew, H. Du, W. Y. B. Lim, Z. Xiong, D. Niyato, C. Miao, and D. I. Kim, "Economics of semantic communication system: An auction approach," IEEE Transactions on Vehicular Technology, 2023.

[39] R. Gozalo-Brizuela and E. C. Garrido-Merchan, "ChatGPT is not all you need. a state of the art review of large generative AI models," arXiv preprint arXiv:2301.04655, Jan. 2023.

[40] Z. Lin, G. Qu, X. Chen, and K. Huang, "Split learning in 6G edge networks," arXiv preprint arXiv:2306.12194, Jun. 2023.

[41] H. Du, J. Liu, D. Niyato, J. Kang, Z. Xiong, J. Zhang, and D. I. Kim, "Attention-aware resource allocation and qoe analysis for metaverse xurllc services," IEEE Journal on Selected Areas in Communications, to appear, 2023.

[42] Q. Yang, Y. Zhao, H. Huang, Z. Xiong, J. Kang, and Z. Zheng, "Fusing blockchain and AI with metaverse: A survey," IEEE Open Journal of the Computer Society, vol. 3, pp. 122-136, Jul. 2022.

[43] X. Ren, M. Xu, D. Niyato, J. Kang, Z. Xiong, C. Qiu, and X. Wang, "Building resilient web 3.0 with quantum information technologies and blockchain: An ambilateral view," arXiv preprint arXiv:2303.13050, Mar. 2023

[44] F. Tiago, F. Moreira, and T. Borges-Tiago, "Youtube videos: A destination marketing outlook," in Proceedings of the Strategic Innovative Marketing and Tourism, Northern Aegean, Greece, May 2019, pp. 877884 .

[45] J. Krumm, N. Davies, and C. Narayanaswami, "User-generated content," IEEE Pervasive Computing, vol. 7, no. 4, pp. 10-11, Oct. 2008.

[46] F.-A. Croitoru, V. Hondru, R. T. Ionescu, and M. Shah, "Diffusion models in vision: A survey," arXiv preprint arXiv:2209.04747, Sep. 2022.

[47] J. Oppenlaender, "Prompt engineering for text-based generative art," arXiv preprint arXiv:2204.13988, Apr. 2022.

[48] G. Marcus, E. Davis, and S. Aaronson, "A very preliminary analysis of DALL-E 2," arXiv preprint arXiv:2204.13807, Apr. 2022.

[49] P.-H. Chi, P.-H. Chung, T.-H. Wu, C.-C. Hsieh, Y.-H. Chen, S.-W. Li, and H.-y. Lee, "Audio ALBERT: A lite BERT for self-supervised
learning of audio representation," in Proceedings of the IEEE Spoken Language Technology Workshop, Shenzhen, China, Jan. 2021, pp. 344350 .

[50] M. Chui, J. Manyika, M. Miremadi, N. Henke, R. Chung, P. Nel, and S. Malhotra, "Notes from the AI frontier: Insights from hundreds of use cases," McKinsey Global Institute, vol. 2, 2018.

[51] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., "Language models are few-shot learners," Advances In Neural Information Processing Systems, vol. 33, pp. 1877-1901, Dec. 2020.

[52] L. Ouyang, J. Wu, X. Jiang, D. Almeida, C. Wainwright, P. Mishkin, C. Zhang, S. Agarwal, K. Slama, A. Ray et al., "Training language models to follow instructions with human feedback," Advances in Neural Information Processing Systems, vol. 35, pp. 27730-27744, Jul. 2022.

[53] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, "Proximal policy optimization algorithms," arXiv preprint arXiv:1707.06347, Jul. 2017.

[54] Q. Dong, L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu, and Z. Sui, "A survey for in-context learning," arXiv preprint arXiv:2301.00234, Jan. 2022.

[55] Microsoft, "Introducing the new Bing," "Accessed Mar. 19, 2023", [Online]. Available: https://www.bing.com/new

[56] J. Spataro, "Introducing Microsoft 365 Copilot - your copilot for work," "Accessed Mar. 19, 2023", [Online]. Available: https://blogs.microsoft.com/blog/2023/03/16/ introducing-microsoft-365-copilot-your-copilot-for-work/

[57] Y. Yang, M. Ma, H. Wu, Q. Yu, P. Zhang, X. You, J. Wu, C. Peng, T.-S. P. Yum, S. Shen et al., " $6 \mathrm{~g}$ network AI architecture for everyonecentric customized services," IEEE Network, pp. 1-10, Jul. 2022.

[58] F. Daniel, P. Kucherbaev, C. Cappiello, B. Benatallah, and M. Allahbakhsh, "Quality control in crowdsourcing: A survey of quality attributes, assessment techniques, and assurance actions," ACM Computing Surveys (CSUR), vol. 51, no. 1, pp. 1-40, Jan. 2018.

[59] H. Zhang, H. Zhang, B. Di, M. Di Renzo, Z. Han, H. V. Poor, and L. Song, "Holographic integrated sensing and communication," IEEE Journal on Selected Areas in Communications, vol. 40, no. 7, pp. 21142130, Mar. 2022.

[60] X. Deng, Y. Jiang, L. T. Yang, M. Lin, L. Yi, and M. Wang, "Data fusion based coverage optimization in heterogeneous sensor networks: A survey," Information Fusion, vol. 52, pp. 90-105, Dec. 2019.

[61] C. Schuhmann, R. Vencu, R. Beaumont, R. Kaczmarczyk, C. Mullis, A. Katta, T. Coombes, J. Jitsev, and A. Komatsuzaki, "LAION-400M: Open dataset of CLIP-filtered 400 million image-text pairs," arXiv preprint arXiv:2111.02114, 2021.

[62] H. Du, J. Wang, D. Niyato, J. Kang, Z. Xiong, J. Zhang, and X. Shen, "Semantic communications for wireless sensing: RIS-aided encoding and self-supervised decoding," IEEE Journal on Selected Areas in Communications, 2023.

[63] S. M. Jain, "Hugging face," in Introduction to Transformers for NLP: With the Hugging Face Library and Models to Solve Problems, 2022, pp. 51-67.

[64] T. Kynkäänniemi, T. Karras, S. Laine, J. Lehtinen, and T. Aila, "Improved precision and recall metric for assessing generative models," Advances In Neural Information Processing Systems, vol. 32, p. 3927-3936, Dec. 2019.

[65] D. H. Park, S. Azadi, X. Liu, T. Darrell, and A. Rohrbach, "Benchmark for compositional text-to-image synthesis," in Proceedings of the Neural Information Processing Systems Datasets and Benchmarks Track, Virtual Conference, Dec. 2021.

[66] C. Wu, S. Yin, W. Qi, X. Wang, Z. Tang, and N. Duan, "Visual ChatGPT: Talking, drawing and editing with visual foundation models," arXiv preprint arXiv:2303.04671, Mar. 2023.

[67] Y. Benny, T. Galanti, S. Benaim, and L. Wolf, "Evaluation metrics for conditional image generation," International Journal of Computer Vision, vol. 129, no. 5, pp. 1712-1731, May 2021.

[68] T. Xu, P. Zhang, Q. Huang, H. Zhang, Z. Gan, X. Huang, and X. He, "Attngan: Fine-grained text to image generation with attentional generative adversarial networks," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Salt Lake City, Utah, Jun. 2018, pp. 1316-1324.

[69] M. F. Naeem, S. J. Oh, Y. Uh, Y. Choi, and J. Yoo, "Reliable fidelity and diversity metrics for generative models," in Proceedings of the International Conference on Machine Learning, Virtual Conference, Nov. 2020, pp. 7176-7185.
[70] H. Du, B. Ma, D. Niyato, J. Kang, Z. Xiong, and Z. Yang, "Rethinking quality of experience for metaverse services: A consumer-based economics perspective," IEEE Network, 2023.

[71] I. Goodfellow, J. Pouget-Abadie, M. Mirza, B. Xu, D. Warde-Farley, S. Ozair, A. Courville, and Y. Bengio, "Generative adversarial networks," Communications of the ACM, vol. 63, no. 11, pp. 139-144, Oct. 2020 .

[72] J. Zhao, M. Mathieu, and Y. LeCun, "Energy-based generative adversarial network," arXiv preprint arXiv:1609.03126, 2016.

[73] D. P. Kingma, M. Welling et al., "An introduction to variational autoencoders," Foundations and Trends® in Machine Learning, vol. 12, no. 4, pp. 307-392, Nov. 2019.

[74] D. Rezende and S. Mohamed, "Variational inference with normalizing flows," in Proceedings of the International Conference on Machine Learning, Lille, France, Jul. 2015, pp. 1530-1538.

[75] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong et al., "A survey of large language models," arXiv preprint arXiv:2303.18223, 2023.

[76] D. Driess, F. Xia, M. S. Sajjadi, C. Lynch, A. Chowdhery, B. Ichter, A. Wahid, J. Tompson, Q. Vuong, T. Yu et al., "PaLM-E: An embodied multimodal language model," arXiv preprint arXiv:2303.03378, 2023.

[77] Z. Chen, Z. Zhang, and Z. Yang, "Big AI models for 6G wireless networks: Opportunities, challenges, and research directions," arXiv preprint arXiv:2308.06250, 2023.

[78] L. Bariah, Q. Zhao, H. Zou, Y. Tian, F. Bader, and M. Debbah, "Large language models for Telecom: The next big thing?" arXiv preprint arXiv:2306.10249, 2023.

[79] Z. Lin, G. Qu, Q. Chen, X. Chen, Z. Chen, and K. Huang, "Pushing large language models to the $6 \mathrm{G}$ edge: Vision, challenges, and opportunities," arXiv preprint arXiv:2309.16739, 2023.

[80] M. Xu, D. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, "Sparks of GPTs in edge intelligence for metaverse: Caching and inference for mobile AIGC services," arXiv preprint arXiv:2304.08782, 2023.

[81] H. T. Dinh, C. Lee, D. Niyato, and P. Wang, "A survey of mobile cloud computing: Architecture, applications, and approaches," Wireless Communications and Mobile Computing, vol. 13, no. 18, pp. 15871611, Oct. 2013.

[82] C. Xu, Y. Ding, C. Chen, Y. Ding, W. Zhou, and S. Wen, "Personalized location privacy protection for location-based services in vehicular networks," IEEE Transactions on Intelligent Transportation Systems, vol. 9, no. 10, pp. 1633-1637, Jul. 2022.

[83] H. Du, J. Zhang, J. Cheng, and B. Ai, "Millimeter wave communications with reconfigurable intelligent surfaces: Performance analysis and optimization," IEEE Transactions on Communications, vol. 69, no. 4, pp. 2752-2768, 2021.

[84] J. Wu, L. Wang, Q. Pei, X. Cui, F. Liu, and T. Yang, "Hitdl: Highthroughput deep learning inference at the hybrid mobile edge," IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 12, pp. 4499-4514, Aug. 2022.

[85] M. Zhang and J. Li, "A commentary of gpt-3 in mit technology review 2021," Fundamental Research, vol. 1, no. 6, pp. 831-833, Feb. 2021.

[86] J. D. M.-W. C. Kenton and L. K. Toutanova, "BERT: Pre-training of deep bidirectional transformers for language understanding," in Proceedings of NAACL-HLT, Minneapolis, Minnesota, Jun. 2019, pp. 4171-4186.

[87] R. Thoppilan, D. De Freitas, J. Hall, N. Shazeer, A. Kulshreshtha, H.T. Cheng, A. Jin, T. Bos, L. Baker, Y. Du et al., "Lamda: Language models for dialog applications," arXiv preprint arXiv:2201.08239, Jan. 2022.

[88] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin, "Attention is all you need," Advances In Neural Information Processing Systems, p. 6000-6010, Dec. 2017.

[89] Y. Zhu, R. Kiros, R. Zemel, R. Salakhutdinov, R. Urtasun, A. Torralba, and S. Fidler, "Aligning books and movies: Towards story-like visual explanations by watching movies and reading books," in Proceedings of the IEEE International Conference on Computer Vision, Santiago, Chile, Dec. 2015, pp. 19-27.

[90] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu, "Bleu: A method for automatic evaluation of machine translation," in Proceedings of the 40th annual meeting of the Association for Computational Linguistics, Philadelphia, Pennsylvania, Jul. 2002, pp. 311-318.

[91] C.-Y. Lin, "ROUGE: A package for automatic evaluation of summaries," in Text summarization branches out, Barcelona, Spain, Jul. 2004, pp. 74-81.

[92] T. Karras, S. Laine, and T. Aila, "A style-based generator architecture for generative adversarial networks," in Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, Long Beach, CA, Jun. 2019, pp. 4401-4410.

[93] A. Brock, J. Donahue, and K. Simonyan, "Large scale GAN training for high fidelity natural image synthesis," arXiv preprint arXiv:1809.11096, Sep. 2018.

[94] A. Sauer, K. Schwarz, and A. Geiger, "Stylegan-x1: Scaling stylegan to large diverse datasets," in Proceedings of the ACM SIGGRAPH, Virtual Conference, Jul. 2022, pp. 1-10.

[95] A. Clark, J. Donahue, and K. Simonyan, "Adversarial video generation on complex datasets," arXiv preprint arXiv:1907.06571, Jul. 2019.

[96] J. Chen, H. Guo, K. Yi, B. Li, and M. Elhoseiny, "Visualgpt: Dataefficient adaptation of pretrained language models for image captioning," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, Virtual Conference, Jun. 2022, pp. 1803018040 .

[97] D. P. Kingma and M. Welling, "Auto-encoding variational bayes," arXiv preprint arXiv:1312.6114, 2013.

[98] C. Saharia, W. Chan, S. Saxena, L. Li, J. Whang, E. L. Denton, K. Ghasemipour, R. Gontijo Lopes, B. Karagol Ayan, T. Salimans et al., "Photorealistic text-to-image diffusion models with deep language understanding," Advances In Neural Information Processing Systems, vol. 35, pp. 36479-36494, Nov. 2022.

[99] J. Sohl-Dickstein, E. Weiss, N. Maheswaranathan, and S. Ganguli, "Deep unsupervised learning using nonequilibrium thermodynamics," in Proceedings of the International Conference on Machine Learning, Lille, France, Jul. 2015, pp. 2256-2265.

[100] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," Advances In Neural Information Processing Systems, vol. 33, pp. 6840-6851, Dec. 2020.

[101] J. Song, C. Meng, and S. Ermon, "Denoising diffusion implicit models," arXiv preprint arXiv:2010.02502, 2020.

[102] I. J. Goodfellow, "On distinguishability criteria for estimating generative models," arXiv preprint arXiv:1412.6515, Dec. 2014.

[103] A. Van Den Oord, O. Vinyals et al., "Neural discrete representation learning," Advances In Neural Information Processing Systems, p. 6309-6318, Dec. 2017.

[104] L. Fei-Fei, J. Deng, and K. Li, "Imagenet: Constructing a large-scale image database," Journal of Vision, vol. 9, no. 8, pp. 1037-1037, Jun. 2009 .

[105] Z. Liu, P. Luo, X. Wang, and X. Tang, "Deep learning face attributes in the wild," in Proceedings of the IEEE International Conference on Computer Vision, Santiago, Chile, Dec. 2015, pp. 3730-3738.

[106] T.-Y. Lin, M. Maire, S. Belongie, J. Hays, P. Perona, D. Ramanan, P. Dollár, and C. L. Zitnick, "Microsoft coco: Common objects in context," in Proceedings of the European Conference on Computer Vision, Zurich, Switzerland, Sep. 2014, pp. 740-755.

[107] M. Heusel, H. Ramsauer, T. Unterthiner, B. Nessler, and S. Hochreiter, "GANs trained by a two time-scale update rule converge to a local Nash equilibrium," Advances In Neural Information Processing Systems, p. 6629-6640, Dec. 2017.

[108] T. Salimans, I. Goodfellow, W. Zaremba, V. Cheung, A. Radford, and X. Chen, "Improved techniques for training GANs," Advances In Neural Information Processing Systems, p. 2234-2242, Dec. 2016.

[109] R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang, "The unreasonable effectiveness of deep features as a perceptual metric," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, Los Alamitos, CA, Jun. 2018, pp. 586-595.

[110] A. Topirceanu, G. Barina, and M. Udrescu, "MuSeNet: Collaboration in the music artists industry," in Proceedings of the European Network Intelligence Conference, Wroclaw, Poland, Sep. 2014, pp. 89-94.

[111] A. van den Oord, S. Dieleman, H. Zen, K. Simonyan, O. Vinyals, A. Graves, N. Kalchbrenner, A. Senior, and K. Kavukcuoglu, "WaveNet: A Generative Model for Raw Audio," in Proceedings of the 9th ISCA Workshop on Speech Synthesis Workshop, Sunnyvale, California, Sep. 2016, p. 125.

[112] Z. Borsos, R. Marinier, D. Vincent, E. Kharitonov, O. Pietquin, M. Sharifi, O. Teboul, D. Grangier, M. Tagliasacchi, and N. Zeghidour, "Audiolm: A language modeling approach to audio generation," arXiv preprint arXiv:2209.03143, Sep. 2022.

[113] C. Hawthorne, A. Stasyuk, A. Roberts, I. Simon, C.-Z. A. Huang, S. Dieleman, E. Elsen, J. Engel, and D. Eck, "Enabling factorized piano music modeling and generation with the MAESTRO dataset," arXiv preprint arXiv:1810.12247, Oct. 2018.
[114] P. Dhariwal and A. Nichol, "Diffusion models beat GANs on image synthesis," Advances In Neural Information Processing Systems, vol. 34, pp. 8780-8794, Dec. 2021.

[115] J. Ho, T. Salimans, A. Gritsenko, W. Chan, M. Norouzi, and D. J. Fleet, "Video diffusion models," arXiv preprint arXiv:2204.03458, Apr. 2022, [Online]. Available: https://arxiv.org/abs/2204.03458

[116] B. Poole, A. Jain, J. T. Barron, and B. Mildenhall, "Dreamfusion: Textto-3D using 2D diffusion," arXiv preprint arXiv:2209.14988, Sep 2022.

[117] W. Kay, J. Carreira, K. Simonyan, B. Zhang, C. Hillier, S. Vijayanarasimhan, F. Viola, T. Green, T. Back, P. Natsev et al., "The kinetics human action video dataset," arXiv preprint arXiv:1705.06950, May 2017.

[118] B. Mildenhall, P. P. Srinivasan, M. Tancik, J. T. Barron, R. Ramamoorthi, and R. Ng, "NeRF: Representing scenes as neural radiance fields for view synthesis," Communications of the ACM, vol. 65, no. 1, pp. 99-106, Jan. 2021.

[119] Z. Lan, M. Chen, S. Goodman, K. Gimpel, P. Sharma, and R. Soricut, "ALBERT: A lite BERT for self-supervised learning of language representations," in Proceedings of the International Conference on Learning Representations, Addis Ababa, Ethiopia, Apr. 2019.

[120] Z. Sun, H. Yu, X. Song, R. Liu, Y. Yang, and D. Zhou, "MobileBERT: a compact task-agnostic BERT for resource-limited devices," in Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, Virtual Conference, Jul. 2020, pp. 2158-2170.

[121] A. Q. Nichol, P. Dhariwal, A. Ramesh, P. Shyam, P. Mishkin, B. Mcgrew, I. Sutskever, and M. Chen, "GLIDE: Towards photorealistic image generation and editing with text-guided diffusion models," in International Conference on Machine Learning, Baltimore, Maryland, Jun. 2022, pp. 16784-16804.

[122] J. Shi, C. Wu, J. Liang, X. Liu, and N. Duan, "DiVAE: Photorealistic images synthesis with denoising diffusion decoder," arXiv preprint arXiv:2206.00386, 2022.

[123] M. Xu, D. Niyato, J. Kang, Z. Xiong, C. Miao, and D. I. Kim, "Wireless edge-empowered metaverse: A learning-based incentive mechanism for virtual reality," in Proceedings of IEEE International Conference on Communications (ICC), Seoul, South Korea, Aug. 2022, pp. 52205225 .

[124] H. Zhang, S. Mao, D. Niyato, and Z. Han, "Location-dependent augmented reality services in wireless edge-enabled metaverse systems," IEEE Open Journal of the Communications Society, vol. 4, pp. 171183, Jan. 2023.

[125] J. Du, F. R. Yu, G. Lu, J. Wang, J. Jiang, and X. Chu, "Mec-assisted immersive vr video streaming over terahertz wireless networks: A deep reinforcement learning approach," IEEE Internet of Things Journal, vol. 7, no. 10, pp. 9517-9529, Jun. 2020.

[126] O. Gafni, A. Polyak, O. Ashual, S. Sheynin, D. Parikh, and Y. Taigman, "Make-a-scene: Scene-based text-to-image generation with human priors," in Proceedings of the 17th European Conference on Computer Vision, Tel Aviv, Israel, 2022, pp. 89-106.

[127] A. Blattmann, R. Rombach, K. Oktay, J. Müller, and B. Ommer, "Semiparametric neural image synthesis," in Advances In Neural Information Processing Systems, Virtual Conference, Nov. 2022.

[128] H. Du, J. Wang, D. Niyato, J. Kang, Z. Xiong, X. S. Shen, and D. I. Kim, "Exploring attention-aware network resource allocation for customized metaverse services," IEEE Network, pp. 1-1, 2022.

[129] W. Jin, N. Ryu, G. Kim, S.-H. Baek, and S. Cho, "Dr. 3D: Adapting 3D GANs to artistic drawings," in Proceedings of the SIGGRAPH Asia, 2022, pp. 1-8.

[130] A. Chen, S. Mao, Z. Li, M. Xu, H. Zhang, D. Niyato, and Z. Han, "An introduction to point cloud compression standards," GetMobile: Mobile Computing and Communications, vol. 27, no. 1, pp. 11-17, May 2023.

[131] G. Chou, Y. Bahat, and F. Heide, "DiffusionSDF: Conditional generative modeling of signed distance functions," arXiv preprint arXiv:2211.13757, Nov. 2022

[132] A. Nichol, H. Jun, P. Dhariwal, P. Mishkin, and M. Chen, "Point-e: A system for generating 3D point clouds from complex prompts," arXiv preprint arXiv:2212.08751, Dec. 2022.

[133] G. Metzer, E. Richardson, O. Patashnik, R. Giryes, and D. Cohen-Or, "Latent-NeRF for shape-guided generation of 3D shapes and textures," arXiv preprint arXiv:2211.07600, Nov. 2022.

[134] X. Zeng, A. Vahdat, F. Williams, Z. Gojcic, O. Litany, S. Fidler, and K. Kreis, "LION: Latent point diffusion models for 3D shape generation," in Advances in Neural Information Processing Systems, Virtual Conference, Nov. 2022.

[135] M. Li, Y. Duan, J. Zhou, and J. Lu, "Diffusion-SDF: Text-to-shape via voxelized diffusion," arXiv preprint arXiv:2212.03293, Dec. 2022.

[136] C.-H. Lin, J. Gao, L. Tang, T. Takikawa, X. Zeng, X. Huang, K. Kreis, S. Fidler, M.-Y. Liu, and T.-Y. Lin, "Magic3d: High-resolution text-to3d content creation," arXiv preprint arXiv:2211.10440, Nov. 2022.

[137] A. N. Wu, R. Stouffs, and F. Biljecki, "Generative adversarial networks in the built environment: A comprehensive review of the application of GANs across data types and scales," Building and Environment, p. 109477, Sep. 2022.

[138] H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, D. I. Kim et al., "Enabling AI-generated content (AIGC) services in wireless edge networks," arXiv preprint arXiv:2301.03220, Jan. 2023.

[139] Z. Li, M. Xu, J. Nie, J. Kang, W. Chen, and S. Xie, "NOMA-enabled cooperative computation offloading for blockchain-empowered Internet of things: A learning approach," IEEE Internet of Things Journal, vol. 8, no. 4, pp. 2364-2378, Aug. 2020.

[140] W.-C. Fan, Y.-C. Chen, D. Chen, Y. Cheng, L. Yuan, and Y.-C. F. Wang, "Frido: Feature pyramid diffusion for complex scene image synthesis," arXiv preprint arXiv:2208.13753, Aug. 2022.

[141] H. Ma, Z. Zhou, X. Zhang, and X. Chen, "Towards carbon-neutral edge computing: Greening edge AI by harnessing spot and future carbon markets," IEEE Internet of Things Journal, pp. 1-1, Apr. 2023.

[142] Y. Lin, H. Du, D. Niyato, J. Nie, J. Zhang, Y. Cheng, and Z. Yang, "Blockchain-aided secure semantic communication for AI-generated content in metaverse," arXiv preprint arXiv:2301.11289, Jan. 2023.

[143] H. Du, J. Wang, D. Niyato, J. Kang, Z. Xiong, and D. I. Kim, "AIgenerated incentive mechanism and full-duplex semantic communications for information sharing," arXiv preprint arXiv:2303.01896, Mar. 2023.

[144] H. Du, Z. Li, D. Niyato, J. Kang, Z. Xiong, H. Huang, and S. Mao, "Generative AI-aided optimization for AI-generated content (AIGC) services in edge networks," arXiv preprint arXiv:2303.13052, 2023.

[145] H. Du, R. Zhang, D. Niyato, J. Kang, Z. Xiong, D. I. Kim, H. V. Poor et al., "Exploring collaborative distributed diffusion-based AI-generated content (aigc) in wireless networks," arXiv preprint arXiv:2304.03446, 2023.

[146] Y. Liu, H. Du, D. Niyato, J. Kang, Z. Xiong, C. Miao, A. Jamalipour et al., "Blockchain-empowered lifecycle management for AIgenerated content (AIGC) products in edge networks," arXiv preprint arXiv:2303.02836, Mar. 2023

[147] Y. Liu, H. Du, D. Niyato, J. Kang, Z. Xiong, D. I. Kim, and A. Jamalipour, "Deep generative model and its applications in efficient wireless network management: A tutorial and case study," arXiv preprint arXiv:2303.17114, 2023.

[148] J. Wang, H. Du, D. Niyato, Z. Xiong, J. Kang, S. Mao et al., "Guiding AI-generated digital content with wireless perception," arXiv preprint arXiv:2303.14624, 2023.

[149] H. Du, D. Niyato, J. Kang, Z. Xiong, K.-Y. Lam, Y. Fang, and Y. Li, "Spear or shield: Leveraging generative AI to tackle security threats of intelligent network services," arXiv preprint arXiv:2306.02384, 2023.

[150] R. Zhang, K. Xiong, H. Du, D. Niyato, J. Kang, X. Shen, and H. V. Poor, "Generative AI-enabled vehicular networks: Fundamentals, framework, and case study," arXiv preprint arXiv:2304.11098, 2023.

[151] Y. Lin, Z. Gao, H. Du, D. Niyato, J. Kang, R. Deng, and X. S. Shen, "A unified blockchain-semantic framework for wireless edge intelligence enabled web 3.0," IEEE Wireless Communications, pp. 1-1, Mar. 2023.

[152] B. Du, H. Du, H. Liu, D. Niyato, P. Xin, J. Yu, M. Qi, and Y. Tang, "YOLO-based semantic communication with generative AIaided resource allocation for digital twins construction," arXiv preprint arXiv:2306.14138, 2023.

[153] M. Xu, D. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, "Joint foundation model caching and inference of generative AI services for edge intelligence," arXiv preprint arXiv:2305.12130, 2023.

[154] M. Xu, D. Niyato, J. Chen, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, "Generative AI-empowered simulation for autonomous driving in vehicular mixed reality metaverses," arXiv preprint arXiv:2302.08418, Feb. 2023.

[155] J. Chen, C. Yi, H. Du, D. Niyato, J. Kang, J. Cai et al., "A revolution of personalized healthcare: Enabling human digital twin with mobile AIGC," arXiv preprint arXiv:2307.12115, 2023.

[156] X. Huang, P. Li, H. Du, J. Kang, D. Niyato, D. I. Kim, and Y. Wu, "Federated learning-empowered AI-generated content in wireless networks," arXiv preprint arXiv:2307.07146, 2023.

[157] J. Wang, H. Du, D. Niyato, J. Kang, S. Cui, X. Shen, and P. Zhang, "Generative AI for integrated sensing and communication: Insights from the physical layer perspective," arXiv preprint arXiv:2310.01036, 2023.
[158] H. Du, G. Liu, D. Niyato, J. Zhang, J. Kang, Z. Xiong, B. Ai, and D. I. Kim, "Generative AI-aided joint training-free secure semantic communications via multi-modal prompts," arXiv preprint arXiv:2309.02616, 2023.

[159] Y. Liu, H. Du, D. Niyato, J. Kang, S. Cui, X. Shen, and P. Zhang, "Optimizing mobile-edge AI-generated everything (AIGX) services by prompt engineering: Fundamental, framework, and case study," arXiv preprint arXiv:2309.01065, 2023.

[160] J. Zheng, J. Zhang, H. Du, D. Niyato, S. Sun, B. Ai, and K. B. Letaief, "Flexible-position MIMO for wireless communications: Fundamentals, challenges, and future directions," arXiv preprint arXiv:2308.14578, 2023.

[161] J. Wang, H. Du, D. Niyato, J. Kang, Z. Xiong, D. Rajan, S. Mao et al., "A unified framework for guiding generative AI with wireless perception in resource-constrained mobile edge networks," arXiv preprint arXiv:2309.01426, 2023.

[162] J. Wang, H. Du, Z. Tian, D. Niyato, J. Kang, and X. Shen, "Semanticaware sensing information transmission for metaverse: A contest theoretic approach," IEEE Transactions on Wireless Communications, pp. $1-1$ Jan. 2023.

[163] J. Wang, H. Du, X. Yang, D. Niyato, J. Kang, and S. Mao, "Wireless sensing data collection and processing for metaverse avatar construction," arXiv preprint arXiv:2211.12720, Nov. 2022.

[164] W. Yang, H. Du, Z. Q. Liew, W. Y. B. Lim, Z. Xiong, D. Niyato, X. Chi, X. S. Shen, and C. Miao, "Semantic communications for future Internet: Fundamentals, applications, and challenges," IEEE Communications Surveys \& Tutorials, vol. 25, no. 1, pp. 213-250, Nov. 2023.

[165] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "High-resolution image synthesis with latent diffusion models," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, Louisiana, Jun. 2022, pp. 10684 10695 .

[166] M. Mathias, R. Timofte, R. Benenson, and L. Van Gool, "Traffic sign recognition-how far are we from the solution?" in Proceedings of the International joint conference on Neural networks, Dallas, Texas, Aug. 2013, pp. 1-8

[167] M. Xu, D. Niyato, B. Wright, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, "Epvisa: Efficient auction design for real-time physical-virtual synchronization in the metaverse," arXiv preprint arXiv:2211.06838, Nov. 2022.

[168] M. Xu, D. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, "Generative AI-empowered effective physical-virtual synchronization in the vehicular metaverse," arXiv preprint arXiv:2301.07636, Jan. 2023.

[169] C. Hu, W. Bao, D. Wang, and F. Liu, "Dynamic adaptive DNN surgery for inference acceleration on the edge," in Proceedings of the IEEE INFOCOM, Paris, France, Apr. 2019, pp. 1423-1431.

[170] W. Zhang, D. Yang, H. Peng, W. Wu, W. Quan, H. Zhang, and X. Shen, "Deep reinforcement learning based resource management for DNN inference in industrial IoT," IEEE Transactions on Vehicular Technology, vol. 70, no. 8, pp. 7605-7618, Mar. 2021.

[171] R. Zhang, K. Xiong, Y. Lu, B. Gao, P. Fan, and K. B. Letaief, "Joint coordinated beamforming and power splitting ratio optimization in mumiso swipt-enabled hetnets: A multi-agent ddqn-based approach," IEEE Journal on Selected Areas in Communications, vol. 40, no. 2, pp. 677693, Oct. 2021.

[172] S. Wang, T. Tuor, T. Salonidis, K. K. Leung, C. Makaya, T. He, and K. Chan, "When edge meets learning: Adaptive control for resourceconstrained distributed machine learning," in Proceedings of the IEEE INFOCOM, Honolulu, HI, Jun. 2018, pp. 63-71.

[173] K. Hsieh, A. Harlap, N. Vijaykumar, D. Konomis, G. R. Ganger, P. B. Gibbons, and O. Mutlu, "Gaia: Geo-distributed machine learning approaching LAN speeds," in Proceedings of the 14th USENIX Symposium on Networked Systems Design and Implementation, Boston, MA, Mar. 2017, pp. 629-647.

[174] Z. Lin, S. Bi, and Y.-J. A. Zhang, "Optimizing AI service placement and resource allocation in mobile edge intelligence systems," IEEE Transactions on Wireless Communications, vol. 20, no. 11, pp. 72577271, May 2021.

[175] X. Li, S. Bi, and H. Wang, "Optimizing resource allocation for joint AI model training and task inference in edge intelligence systems," IEEE Wireless Communications Letters, vol. 10, no. 3, pp. 532-536, Mar. 2020 .

[176] K. Zhao, Z. Zhou, X. Chen, R. Zhou, X. Zhang, S. Yu, and D. Wu, "EdgeAdaptor: Online configuration adaption, model selection and
resource provisioning for edge DNN inference serving at scale," IEEE Transactions on Mobile Computing, pp. 1-16, Jul. 2022.

[177] X. Tang, X. Chen, L. Zeng, S. Yu, and L. Chen, "Joint multiuser DNN partitioning and computational resource allocation for collaborative edge intelligence," IEEE Internet of Things Journal, vol. 8, no. 12, pp. 9511-9522, Jul. 2020.

[178] W. Y. B. Lim, J. S. Ng, Z. Xiong, J. Jin, Y. Zhang, D. Niyato, C. Leung, and C. Miao, "Decentralized edge intelligence: A dynamic resource allocation framework for hierarchical federated learning," IEEE Transactions on Parallel and Distributed Systems, vol. 33, no. 3, pp. 536-550, Jul. 2021.

[179] Y. Yang, Z. Zhang, Y. Tian, Z. Yang, C. Huang, C. Zhong, and K.-K. Wong, "Over-the-air split machine learning in wireless MIMO networks," IEEE Journal on Selected Areas in Communications, vol. 41, no. 4, pp. 1007-1022, Feb. 2023.

[180] R. Zhang, K. Xiong, Y. Lu, P. Fan, D. W. K. Ng, and K. B. Letaief, "Energy efficiency maximization in ris-assisted swipt networks with rsma: A ppo-based approach," IEEE Journal on Selected Areas in Communications, pp. 1-1, Jan. 2023.

[181] G. Ditzler, M. Roveri, C. Alippi, and R. Polikar, "Learning in nonstationary environments: A survey," IEEE Computational Intelligence Magazine, vol. 10, no. 4, pp. 12-25, Nov. 2015.

[182] R. Zhang, K. Xiong, X. Tian, Y. Lu, P. Fan, and K. B. Letaief, "Inverse reinforcement learning meets power allocation in multi-user cellular networks," in IEEE INFOCOM 2022-IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), New York, NY, May 2022, pp. 1-2.

[183] W. Wu, Y. Tang, P. Yang, W. Zhang, and N. Zhang, "Collaborative deep neural network inference via mobile edge computing," in Broadband Communications, Computing, and Control for Ubiquitous Intelligence. Springer, Mar. 2022, pp. 263-290.

[184] W. Fan, Z. Chen, Y. Su, F. Wu, B. Tang, and Y. Liu, "Accuracy-based task offloading and resource allocation for edge intelligence in IoT," IEEE Wireless Communications Letters, vol. 11, no. 2, pp. 371-375, Nov. 2021.

[185] X. Chen, M. Li, H. Zhong, Y. Ma, and C.-H. Hsu, "DNNOff: offloading DNN-based intelligent IoT applications in mobile edge computing," IEEE Transactions on Industrial Informatics, vol. 18, no. 4, pp. 28202829, Apr. 2021.

[186] B. Lin, Y. Huang, J. Zhang, J. Hu, X. Chen, and J. Li, "Cost-driven offloading for DNN-based applications over cloud, edge, and end devices," IEEE Transactions on Industrial Informatics, vol. 16, no. 8, pp. 54565466, Aug. 2019.

[187] L. Ren, Y. Laili, X. Li, and X. Wang, "Coding-based large-scale task assignment for industrial edge intelligence," IEEE Transactions on Network Science and Engineering, vol. 7, no. 4, pp. 2286-2297, Sep. 2019.

[188] H.-J. Jeong, I. Jeong, H.-J. Lee, and S.-M. Moon, "Computation offloading for machine learning web apps in the edge server environment," in Proceedings of the IEEE 38th International Conference on Distributed Computing Systems, Vienna, Austria, Jul. 2018, pp. 14921499 .

[189] X. Li, C. Sun, J. Wen, X. Wang, M. Guizani, and V. C. Leung, "Multiuser qoe enhancement: Federated multi-agent reinforcement learning for cooperative edge intelligence," IEEE Network, vol. 36, no. 5, pp. 144-151, Nov. 2022.

[190] Y. Zhan, S. Guo, P. Li, and J. Zhang, "A deep reinforcement learningbased offloading game in edge computing," IEEE Transactions on Computers, vol. 69, no. 6, pp. 883-893, Jan. 2020.

[191] Z. Lin, G. Zhu, Y. Deng, X. Chen, Y. Gao, K. Huang, and Y. Fang, "Efficient parallel split learning over resource-constrained wireless edge networks," arXiv preprint arXiv:2303.15991, Mar. 2023.

[192] Y.-C. Wang, J. Xue, C. Wei, and C.-C. J. Kuo, "An overview on generative AI at scale with edge-cloud computing," arXiv preprint arXiv:2306.17170, Jun. 2023.

[193] W. Wu, P. Yang, W. Zhang, C. Zhou, and X. Shen, "Accuracyguaranteed collaborative DNN inference in industrial IoT via deep reinforcement learning," IEEE Transactions on Industrial Informatics, vol. 17, no. 7, pp. 4988-4998, Aug. 2020.

[194] Z. Yang, M. Chen, K.-K. Wong, H. V. Poor, and S. Cui, "Federated learning for 6G: Applications, challenges, and opportunities," Engineering, vol. 8, pp. 33-41, Jan. 2022.

[195] Y. Tian, Z. Zhang, Z. Yang, and Q. Yang, "Jmsnas: Joint model split and neural architecture search for learning over mobile edge networks," in 2022 IEEE International Conference on Communications Workshops (ICC Workshops), Seoul, South Korea, May 2022, pp. 103-108.
[196] R. Zhang, K. Xiong, W. Guo, X. Yang, P. Fan, and K. B. Letaief, "Qlearning-based adaptive power control in wireless RF energy harvesting heterogeneous networks," IEEE Systems Journal, vol. 15, no. 2, pp. 1861-1872, Sep. 2020.

[197] D. Wen, X. Jiao, P. Liu, G. Zhu, Y. Shi, and K. Huang, "Task-oriented over-the-air computation for multi-device edge split inference," in 2023 IEEE Wireless Communications and Networking Conference (WCNC), Glasgow, United Kingdom, Mar. 2023, pp. 1-6.

[198] Y. Koda, J. Park, M. Bennis, K. Yamamoto, T. Nishio, M. Morikura, and K. Nakashima, "Communication-efficient multimodal split learning for mmwave received power prediction," IEEE Communications Letters, vol. 24, no. 6, pp. 1284-1288, Mar. 2020.

[199] W. Wu, M. Li, K. Qu, C. Zhou, X. Shen, W. Zhuang, X. Li, and W. Shi, "Split learning over wireless networks: Parallel design and resource management," IEEE Journal on Selected Areas in Communications, vol. 41, no. 4, pp. 1051-1066, Feb. 2023.

[200] D. Saguil and A. Azim, "A layer-partitioning approach for faster execution of neural network-based embedded applications in edge networks," IEEE Access, vol. 8, pp. 59 456-59 469, Mar. 2020.

[201] J. Kang, H. Du, Z. Li, Z. Xiong, S. Ma, D. Niyato, and Y. Li, "Personalized saliency in task-oriented semantic communications: Image transmission and performance analysis," IEEE Journal on Selected Areas in Communications, vol. 41, no. 1, pp. 186-201, 2022.

[202] Z. Yang, R. Wang, D. Wu, H. Wang, H. Song, and X. Ma, "Local trajectory privacy protection in $5 \mathrm{G}$ enabled industrial intelligent logistics," IEEE Transactions on Industrial Informatics, vol. 18, no. 4, pp. 2868-2876, Sep. 2021.

[203] W. Zhang, D. Yang, Y. Xu, X. Huang, J. Zhang, and M. Gidlund, "Deephealth: A self-attention based method for instant intelligent predictive maintenance in industrial Internet of things," IEEE Transactions on Industrial Informatics, vol. 17, no. 8, pp. 5461-5473, Oct. 2020.

[204] W. Zhang, D. Yang, W. Wu, H. Peng, N. Zhang, H. Zhang, and X. Shen, "Optimizing federated learning in distributed industrial IoT: A multiagent approach," IEEE Journal on Selected Areas in Communications, vol. 39, no. 12, pp. 3688-3703, Oct. 2021.

[205] Y. Matsubara, S. Baidya, D. Callegaro, M. Levorato, and S. Singh, "Distilled split deep neural networks for edge-assisted real-time systems," in Proceedings of the 2019 Workshop on Hot Topics in Video Analytics and Intelligent Edges, Los Cabos, Mexico, Oct. 2019, pp. 21-26.

[206] K. Jiang, C. Sun, H. Zhou, X. Li, M. Dong, and V. C. Leung, "Intelligence-empowered mobile edge computing: Framework, issues, implementation, and outlook," IEEE Network, vol. 35, no. 5, pp. 74-82, Nov. 2021

[207] C. Sun, X. Wu, X. Li, Q. Fan, J. Wen, and V. C. Leung, "Cooperative computation offloading for multi-access edge computing in 6G mobile networks via soft actor critic," IEEE Transactions on Network Science and Engineering, pp. 1-1, Apr. 2021.

[208] X. He, K. Wang, H. Lu, W. Xu, and S. Guo, "Edge QoE: Intelligent big data caching via deep reinforcement learning," IEEE Network, vol. 34, no. 4, pp. 8-13, Jul. 2020.

[209] T. Guo, R. J. Walls, and S. S. Ogden, "Edgeserve: efficient deep learning model caching at the edge," in Proceedings of the 4th ACM/IEEE Symposium on Edge Computing, Arlington, Virginia, Nov. 2019, pp. 313-315.

[210] S. S. Ogden, G. R. Gilman, R. J. Walls, and T. Guo, "Many models at the edge: Scaling deep inference via model-level caching," in Proceedings of the IEEE International Conference on Autonomic Computing and Self-Organizing Systems, Washington, DC, Sep. 2021, pp. 51-60.

[211] M. Xu, M. Zhu, Y. Liu, F. X. Lin, and X. Liu, "Deepcache: Principled cache for mobile deep vision," in Proceedings of the 24th Annual International Conference on Mobile Computing and Networking, New Delhi, India, Oct. 2018, pp. 129-144.

[212] A. Fuerst and P. Sharma, "Faascache: keeping serverless computing alive with greedy-dual caching," in Proceedings of the 26th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Virtual Conference, Mar. 2021, pp. 386-400.

[213] X.-Y. Zheng, M.-C. Lee, and Y.-W. P. Hong, "Knowledge caching for federated learning," in 2021 IEEE Global Communications Conference (GLOBECOM). IEEE, 2021, pp. 1-6.

[214] X. Wang, R. Li, C. Wang, X. Li, T. Taleb, and V. C. Leung, "Attentionweighted federated deep reinforcement learning for device-to-device assisted heterogeneous collaborative edge caching," IEEE Journal on Selected Areas in Communications, vol. 39, no. 1, pp. 154-169, Nov. 2020 .

[215] Y. Mu and C. Shen, "Communication and storage efficient federated split learning," arXiv preprint arXiv:2302.05599, Feb. 2023.

[216] M. Yao, L. Chen, J. Zhang, J. Huang, and J. Wu, "Loading cost-aware model caching and request routing for cooperative edge inference," in Proceedings of the IEEE International Conference on Communication, Seoul, South Korea, May 2022, pp. 2327-2332.

[217] Y. Shi, K. Yang, T. Jiang, J. Zhang, and K. B. Letaief, "Communicationefficient edgeAI: Algorithms and systems," IEEE Communications Surveys \& Tutorials, vol. 22, no. 4, pp. 2167-2191, Jul. 2020.

[218] S. Xie, Y. Wu, S. Ma, M. Ding, Y. Shi, and M. Tang, "Robust information bottleneck for task-oriented communication with digital modulation," arXiv preprint arXiv:2209.10382, Sep. 2022.

[219] S. S. Ogden and T. Guo, "Mdinference: Balancing inference accuracy and latency for mobile applications," in Proceedings of the IEEE International Conference on Cloud Engineering, NSW, Australia, Apr. 2020, pp. 28-39

[220] M. Buckler, P. Bedoukian, S. Jayasuriya, and A. Sampson, "Eva 2 : Exploiting temporal redundancy in live computer vision," in 2018 ACM/IEEE 45th Annual International Symposium on Computer Architecture (ISCA), Los Angeles, California, Jun. 2018, pp. 533-546.

[221] E. Oakes, L. Yang, D. Zhou, K. Houck, T. Harter, A. Arpaci-Dusseau, and R. Arpaci-Dusseau, " $\{$ SOCK\}: Rapid task provisioning with serverless-optimized containers," in Proceedings of the \{USENIX\} Annual Technical Conference ( $\{$ USENIX\} $\{$ ATC $\}$ 18), Boston, MA, Jul. 2018, pp. 57-70.

[222] M. Chen, Z. Yang, W. Saad, C. Yin, H. V. Poor, and S. Cui, "A joint learning and communications framework for federated learning over wireless networks," IEEE Transactions on Wireless Communications, vol. 20, no. 1, pp. 269-283, Oct. 2020 .

[223] M. Xu, D. Niyato, Z. Yang, Z. Xiong, J. Kang, D. I. Kim, and X. Shen, "Privacy-preserving intelligent resource allocation for federated edge learning in quantum Internet," IEEE Journal of Selected Topics in Signal Processing, vol. 17, no. 1, pp. 142-157, Nov. 2023.

[224] M. Chen, N. Shlezinger, H. V. Poor, Y. C. Eldar, and S. Cui, "Communication-efficient federated learning," Proceedings of the National Academy of Sciences, vol. 118, no. 17, p. e2024789118, Apr. 2021.

[225] Q. Wang, Z. Li, K. Nai, Y. Chen, and M. Wen, "Dynamic resource allocation for jointing vehicle-edge deep neural network inference," Journal of Systems Architecture, vol. 117, p. 102133, Aug. 2021.

[226] K. Yang, P. Sun, J. Lin, A. Boukerche, and L. Song, "A novel distributed task scheduling framework for supporting vehicular edge intelligence," in Proceedings of the IEEE 42nd International Conference on Distributed Computing Systems, Bologna, Italy, Jul. 2022, pp. 972-982.

[227] Y. Sun, B. Xie, S. Zhou, and Z. Niu, "MEET: Mobility-enhanced edge intelligence for smart and green 6G networks," IEEE Communications Magazine, vol. 61, no. 1, pp. 64-70, Oct. 2023.

[228] D. Wang, B. Song, P. Lin, F. R. Yu, X. Du, and M. Guizani, "Resource management for edge intelligence (EI)-assisted IoV using quantuminspired reinforcement learning," IEEE Internet of Things Journal, vol. 9, no. 14, pp. 12588-12600, Dec. 2021

[229] V. Balasubramanian, S. Otoum, and M. Reisslein, "Venet: hybrid stacked autoencoder learning for cooperative edge intelligence in IoV," IEEE Transactions on Intelligent Transportation Systems, vol. 23, no. 9, pp. 16643-16653, May 2022

[230] C. Dong, Y. Shen, Y. Qu, K. Wang, J. Zheng, Q. Wu, and F. Wu, "UAVs as an intelligent service: Boosting edge intelligence for airground integrated networks," IEEE Network, vol. 35, no. 4, pp. 167175, Aug. 2021.

[231] H. Luo, T. Chen, X. Li, S. Li, C. Zhang, G. Zhao, and X. Liu, "KeepEdge: A knowledge distillation empowered edge intelligence framework for visual assisted positioning in UAV delivery," IEEE Transactions on Mobile Computing, pp. 1-1, Mar. 2022.

[232] S. Zhou, Y. Sun, Z. Jiang, and Z. Niu, "Exploiting moving intelligence: Delay-optimized computation offloading in vehicular fog networks," IEEE Communications Magazine, vol. 57, no. 5, pp. 49-55, May 2019.

[233] L. Zhu, J. Zhang, Z. Xiao, X. Cao, X.-G. Xia, and R. Schober, "Millimeter-wave full-duplex UAV relay: Joint positioning, beamforming, and power control," IEEE Journal on Selected Areas in Communications, vol. 38, no. 9, pp. 2057-2073, 2020.

[234] H. Du, D. Niyato, Y.-A. Xie, Y. Cheng, J. Kang, and D. I. Kim, "Performance analysis and optimization for jammer-aided multiantenna UAV covert communication," IEEE Journal on Selected Areas in Communications, vol. 40, no. 10, pp. 2962-2979, Oct. 2022.

[235] J. Kang, H. Du, Z. Li, Z. Xiong, S. Ma, D. Niyato, and Y. Li, "Personalized saliency in task-oriented semantic communications: Image transmission and performance analysis," IEEE Journal on Selected Areas in Communications, vol. 41, no. 1, pp. 186-201, Nov. 2023.

[236] L. N. Huynh and E.-N. Huh, "UAV-enhanced edge intelligence: A survey," in Proceedings of the 6th International Conference on Computing Methodologies and Communication, Erode, India, Mar. 2022, pp. 4247.

[237] S. H. Alsamhi, F. A. Almalki, F. Afghah, A. Hawbani, A. V. Shvetsov, B. Lee, and H. Song, "Drones" edge intelligence over smart environments in B5G: Blockchain and federated learning synergy," IEEE Transactions on Green Communications and Networking, vol. 6, no. 1, pp. 295-312, Dec. 2021.

[238] Z. Wang, Y. Zhou, Y. Shi, and W. Zhuang, "Interference management for over-the-air federated learning in multi-cell wireless networks," IEEE Journal on Selected Areas in Communications, vol. 40, no. 8, pp. 2361-2377, Jun. 2022.

[239] T. Yang, S. Gao, J. Li, M. Qin, X. Sun, R. Zhang, M. Wang, and X. Li, "Multi-armed bandits learning for task offloading in maritime edge intelligence networks," IEEE Transactions on Vehicular Technology, vol. 71, no. 4, pp. 4212-4224, Jan. 2022.

[240] Z. Wang, J. Qiu, Y. Zhou, Y. Shi, L. Fu, W. Chen, and K. B. Letaief, "Federated learning via intelligent reflecting surface," IEEE Transactions on Wireless Communications, vol. 21, no. 2, pp. 808822, Jul. 2021

[241] W. Quan, N. Cheng, M. Qin, H. Zhang, H. A. Chan, and X. Shen, "Adaptive transmission control for software defined vehicular networks," IEEE Wireless Communications Letters, vol. 8, no. 3, pp. 653656, Nov. 2018.

[242] S. Misra and S. Bera, "Soft-VAN: Mobility-aware task offloading in software-defined vehicular network," IEEE Transactions on Vehicular Technology, vol. 69, no. 2, pp. 2071-2078, Dec. 2019

[243] Y. Sun, W. Shi, X. Huang, S. Zhou, and Z. Niu, "Edge learning with timeliness constraints: Challenges and solutions," IEEE Communications Magazine, vol. 58, no. 12, pp. 27-33, Dec. 2020.

[244] J. Wang, K. Zhu, and E. Hossain, "Green Internet of vehicles (IoV) in the $6 \mathrm{G}$ era: Toward sustainable vehicular communications and networking," IEEE Transactions on Green Communications and Networking, vol. 6, no. 1, pp. 391-423, Nov. 2021.

[245] X. Huang, P. Li, R. Yu, Y. Wu, K. Xie, and S. Xie, "Fedparking: A federated learning based parking space estimation with parked vehicle assisted edge computing," IEEE Transactions on Vehicular Technology, vol. 70, no. 9, pp. 9355-9368, Jul. 2021.

[246] M. Xu, D. T. Hoang, J. Kang, D. Niyato, Q. Yan, and D. I. Kim, "Secure and reliable transfer learning framework for 6G-enabled Internet of vehicles," IEEE Wireless Communications, vol. 29, no. 4, pp. 132-139, May 2022.

[247] M. Li, J. Gao, L. Zhao, and X. Shen, "Deep reinforcement learning for collaborative edge computing in vehicular networks," IEEE Transactions on Cognitive Communications and Networking, vol. 6, no. 4, pp. 1122-1135, Jun. 2020.

[248] D. Wu, T. Liu, Z. Li, T. Tang, and R. Wang, "Delay-aware edgeterminal collaboration in green Internet of vehicles: A multi-agent soft actor-critic approach," IEEE Transactions on Green Communications and Networking, pp. 1-1, Jun. 2022.

[249] M. Wu, G. Cheng, P. Li, R. Yu, Y. Wu, M. Pan, and R. Lu, "Split learning with differential privacy for integrated terrestrial and nonterrestrial networks," IEEE Wireless Communications, pp. 1-1, Apr. 2023.

[250] J. Yao, "Split learning for image classification in Internet of drones networks," in 2023 IEEE 24th International Conference on High Performance Switching and Routing (HPSR), Albuquerque, NM, Jun. 2023, pp. 52-55.

[251] Y. Zhan and J. Zhang, "An incentive mechanism design for efficient edge learning by deep reinforcement learning approach," in Proceedings of the IEEE INFOCOM, ON, Canada, Jul. 2020, pp. 2489-2498.

[252] Y. Liu, L. Wu, Y. Zhan, S. Guo, and Z. Hong, "Incentive-driven long-term optimization for edge learning by hierarchical reinforcement mechanism," in Proceedings of IEEE 41st International Conference on Distributed Computing Systems, DC, USA, Jul. 2021, pp. 35-45.

[253] Y. Deng, F. Lyu, J. Ren, Y.-C. Chen, P. Yang, Y. Zhou, and Y. Zhang, "Fair: Quality-aware federated learning with precise user incentive and model aggregation," in Proceedings of the IEEE INFOCOM, BC, Canada, May 2021, pp. 1-10.

[254] X. Ren, C. Qiu, X. Wang, Z. Han, K. Xu, H. Yao, and S. Guo, "AI-Bazaar: A cloud-edge computing power trading framework for ubiquitous AI services," IEEE Transactions on Cloud Computing, pp. $1-1$, Aug. 2022.

[255] X. Wang, Y. Zhao, C. Qiu, Z. Liu, J. Nie, and V. C. Leung, "Infedge: A blockchain-based incentive mechanism in hierarchical federated learning for end-edge-cloud communications," IEEE Journal on Selected Areas in Communications, vol. 40, no. 12, pp. 3325-3342, Oct. 2022.

[256] Y. Zhan, J. Zhang, Z. Hong, L. Wu, P. Li, and S. Guo, "A survey of incentive mechanism design for federated learning," IEEE Transactions on Emerging Topics in Computing, vol. 10, no. 2, pp. 1035-1044, Mar. 2021.

[257] H. Du, J. Kang, D. Niyato, J. Zhang, and D. I. Kim, "Reconfigurable intelligent surface-aided joint radar and covert communications: Fundamentals, optimization, and challenges," IEEE Vehicular Technology Magazine, vol. 17, no. 3, pp. 54-64, 2022.

[258] X. Chen, Y. Deng, G. Zhu, D. Wang, and Y. Fang, "From resource auction to service auction: An auction paradigm shift in wireless networks," IEEE Wireless Communications, vol. 29, no. 2, pp. 185191, Apr. 2022.

[259] L. Wu, S. Guo, Y. Liu, Z. Hong, Y. Zhan, and W. Xu, "Sustainable federated learning with long-term online vcg auction mechanism," in Proceedings of the IEEE 42nd International Conference on Distributed Computing Systems. Bologna, Italy: IEEE, Jul. 2022, pp. 895-905.

[260] Y. Zhan, P. Li, Z. Qu, D. Zeng, and S. Guo, "A learning-based incentive mechanism for federated learning," IEEE Internet of Things Journal, vol. 7, no. 7, pp. 6360-6368, Jan. 2020.

[261] J. Du, W. Cheng, G. Lu, H. Cao, X. Chu, Z. Zhang, and J. Wang, "Resource pricing and allocation in mec enabled blockchain systems: An a3c deep reinforcement learning approach," IEEE Transactions on Network Science and Engineering, vol. 9, no. 1, pp. 33-44, Mar. 2021.

[262] J. Ren, G. Yu, Y. He, and G. Y. Li, "Collaborative cloud and edge computing for latency minimization," IEEE Transactions on Vehicular Technology, vol. 68, no. 5, pp. 5031-5044, Mar. 2019.

[263] Z. Tian, L. Cui, J. Liang, and S. Yu, "A comprehensive survey on poisoning attacks and countermeasures in machine learning," ACM Computing Surveys, vol. 55, no. 8, pp. 1-35, Dec. 2022.

[264] Q. Liu, P. Li, W. Zhao, W. Cai, S. Yu, and V. C. Leung, "A survey on security threats and defensive techniques of machine learning: A data-driven view," IEEE Access, vol. 6, pp. 12 103-12 117, Feb. 2018.

[265] L. Xue, J. Ni, D. Liu, X. Lin, and X. Shen, "Blockchain-based fair and fine-grained data trading with privacy preservation," IEEE Transactions on Computers, pp. 1-1, Mar. 2023.

[266] C. Chen, Z. Wu, Y. Lai, W. Ou, T. Liao, and Z. Zheng, "Challenges and remedies to privacy and security in AIGC: Exploring the potential of privacy computing, blockchain, and beyond," arXiv preprint arXiv:2306.00419, Jun. 2023.

[267] J. Kang, J. He, H. Du, Z. Xiong, Z. Yang, X. Huang, and S. Xie, "Adversarial attacks and defenses for semantic communication in vehicular metaverses," arXiv preprint arXiv:2306.03528, Jun. 2023.

[268] S. Zhang, W. Wu, P. Hu, S. Li, and N. Zhang, "Split federated learning: Speed up model training in resource-limited wireless networks," arXiv preprint arXiv:2305.18889, May 2023.

[269] J. Li, Y. Meng, L. Ma, S. Du, H. Zhu, Q. Pei, and X. Shen, "A federated learning based privacy-preserving smart healthcare system," IEEE Transactions on Industrial Informatics, vol. 18, no. 3, pp. 20212031, Jul. 2021.

[270] Z. Wang, G. Yang, H. Dai, and C. Rong, "Privacy-preserving split learning for large-scaled vision pre-training," IEEE Transactions on Information Forensics and Security, vol. 18, pp. 1539-1553, Feb. 2023.

[271] X. Liu, Y. Deng, and T. Mahmoodi, "Wireless distributed learning: a new hybrid split and federated learning approach," IEEE Transactions on Wireless Communications, vol. 22, no. 4, pp. 2650-2665, Oct. 2022.

[272] J. Kang, D. Ye, J. Nie, J. Xiao, X. Deng, S. Wang, Z. Xiong, R. Yu, and D. Niyato, "Blockchain-based federated learning for industrial metaverses: Incentive scheme with optimal aoi," in 2022 IEEE International Conference on Blockchain (Blockchain), Espoo, Finland, Aug. 2022, pp. 71-78.

[273] J. Kang, X. Li, J. Nie, Y. Liu, M. Xu, Z. Xiong, D. Niyato, and Q. Yan, "Communication-efficient and cross-chain empowered federated learning for artificial intelligence of things," IEEE Transactions on Network Science and Engineering, vol. 9, no. 5, pp. 2966-2977, May 2022.

[274] L. Cui, Y. Qu, G. Xie, D. Zeng, R. Li, S. Shen, and S. Yu, "Security and privacy-enhanced federated learning for anomaly detection in IoT infrastructures," IEEE Transactions on Industrial Informatics, vol. 18, no. 5, pp. 3492-3500, Aug. 2021.

[275] S. Augenstein, H. B. McMahan, D. Ramage, S. Ramaswamy, P. Kairouz, M. Chen, R. Mathews et al., "Generative models for effective ML on private, decentralized datasets," arXiv preprint arXiv:1911.06679, Nov. 2019.
[276] C. Fan and P. Liu, "Federated generative adversarial learning," in Proceedings of the Pattern Recognition and Computer Vision, Nanjing, China, Oct. 2020, pp. 3-15.

[277] J. Chung, K. Lee, and K. Ramchandran, "Federated unsupervised clustering with generative models," in Proceedings of the AAAI International Workshop on Trustable, Verifiable and Auditable Federated Learning, 2022.

[278] Z. Wang, Y. Hu, J. Xiao, and C. Wu, "Efficient ring-topology decentralized federated learning with deep generative models for industrial artificial intelligent," Electronics, vol. 11, no. 10, p. 1548, May 2022.

[279] S. Shen, Y. Ren, Y. Ju, X. Wang, W. Wang, and V. C. Leung, "Edgematrix: A resource-redefined scheduling framework for sla-guaranteed multi-tier edge-cloud computing systems," IEEE Journal on Selected Areas in Communications, vol. 41, no. 3, pp. 820-834, Dec. 2023.

[280] K. Gai, J. Guo, L. Zhu, and S. Yu, "Blockchain meets cloud computing: A survey," IEEE Communications Surveys \& Tutorials, vol. 22, no. 3, pp. 2009-2030, Apr. 2020.

[281] Y. Lin, Z. Gao, Y. Tu, H. Du, D. Niyato, J. Kang, and H. Yang, "A blockchain-based semantic exchange framework for web 3.0 toward participatory economy," arXiv preprint arXiv:2211.16662, Nov. 2022.

[282] Y. Lin, Z. Gao, W. Shi, Q. Wang, H. Li, M. Wang, Y. Yang, and L. Rui, "A novel architecture combining oracle with decentralized learning for IIoT," IEEE Internet of Things Journal, vol. 10, no. 5, pp. 3774-3785, Mar. 2023.

[283] C. Huang, W. Wang, D. Liu, R. Lu, and X. Shen, "Blockchainassisted personalized car insurance with privacy preservation and fraud resistance," IEEE Transactions on Vehicular Technology, vol. 72, no. 3, pp. 3777-3792, Mar. 2023.

[284] M. Shen, X. Tang, L. Zhu, X. Du, and M. Guizani, "Privacy-preserving support vector machine training over blockchain-based encrypted IoT data in smart cities," IEEE Internet of Things Journal, vol. 6, no. 5, pp. 7702-7712, Feb. 2019.

[285] M. Shen, H. Lu, F. Wang, H. Liu, and L. Zhu, "Secure and efficient blockchain-assisted authentication for edge-integrated Internet-ofvehicles," IEEE Transactions on Vehicular Technology, vol. 71, no. 11, pp. $12250-12263$, Jul. 2022.

[286] M. Shen, H. Liu, L. Zhu, K. Xu, H. Yu, X. Du, and M. Guizani, "Blockchain-assisted secure device authentication for cross-domain industrial IoT," IEEE Journal on Selected Areas in Communications, vol. 38, no. 5, pp. 942-954, Mar. 2020.

[287] M. Xu, X. Ren, D. Niyato, J. Kang, C. Qiu, Z. Xiong, X. Wang, and V. Leung, "When quantum information technologies meet blockchain in web 3.0," arXiv preprint arXiv:2211.15941, Nov. 2022.

[288] Y. Lin, J. Kang, D. Niyato, Z. Gao, and Q. Wang, "Efficient consensus and elastic resource allocation empowered blockchain for vehicular networks," IEEE Transactions on Vehicular Technology, pp. 1-6, Dec. 2022.

[289] K. P. Dirgantoro, J. M. Lee, and D.-S. Kim, "Generative adversarial networks based on edge computing with blockchain architecture for security system," in Proceedings of the International Conference on Artificial Intelligence in Information and Communication, Fukuoka, Japan, Feb. 2020, pp. 039-042.

[290] W. J.-W. Tann, A. Vuputuri, and E.-C. Chang, "Predicting non-fungible token (NFT) collections: A contextual generative approach," arXiv preprint arXiv:2210.15493, Oct. 2022.

[291] Y. Li, C. Chen, N. Liu, H. Huang, Z. Zheng, and Q. Yan, "A blockchainbased decentralized federated learning framework with committee consensus," IEEE Network, vol. 35, no. 1, pp. 234-241, Dec. 2020.

[292] Y. Liu, Y. Lan, B. Li, C. Miao, and Z. Tian, "Proof of learning (pole): Empowering neural network training with consensus building on blockchains," Computer Networks, vol. 201, p. 108594, Dec. 2021.

[293] S. Zhang, M. Xu, W. Y. B. Lim, and D. Niyato, "Sustainable AIGC workload scheduling of geo-Distributed data centers: A multi-agent reinforcement learning approach," arXiv preprint arXiv:2304.07948, Apr. 2023

[294] H. Ma, R. Li, X. Zhang, Z. Zhou, and X. Chen, "Reliability-aware online scheduling for DNN inference tasks in mobile edge computing," IEEE Internet of Things Journal, pp. 1-1, Feb. 2023.

[295] Z. Wang, J. Zhang, B. Ai, C. Yuen, and M. Debbah, "Uplink performance of cell-free massive MIMO with multi-antenna users over jointly-correlated rayleigh fading channels," IEEE Transactions on Wireless Communications, vol. 21, no. 9, pp. 7391-7406, 2022.

[296] Z. Wang, J. Zhang, H. Q. Ngo, B. Ai, and M. Debbah, "Uplink precoding design for cell-free massive MIMO with iteratively weighted mmse," IEEE Transactions on Communications, vol. 71, no. 3, pp. $1646-1664,2023$.

[297] L. Zhu, J. Zhang, Z. Xiao, X.-G. Xia, and R. Zhang, "Multi-uav aided millimeter-wave networks: Positioning, clustering, and beamforming," IEEE Transactions on Wireless Communications, vol. 21, no. 7, pp. 4637-4653, 2021 .

[298] Y. Shi, Y. Zhou, D. Wen, Y. Wu, C. Jiang, and K. B. Letaief, "Taskoriented communications for 6G: Vision, principles, and technologies," arXiv preprint arXiv:2303.10920, Mar. 2023.

[299] Y. Cheng, D. Wang, P. Zhou, and T. Zhang, "Model compression and acceleration for deep neural networks: The principles, progress, and challenges," IEEE Signal Processing Magazine, vol. 35, no. 1, pp. 126136, Jan. 2018

[300] Z. Li, W. Su, M. Xu, R. Yu, D. Niyato, and S. Xie, "Compact learning model for dynamic off-chain routing in blockchain-based IoT," IEEE Journal on Selected Areas in Communications, vol. 40, no. 12, pp. 3615-3630, Oct. 2022.

[301] Y. Huang, M. Xu, X. Zhang, D. Niyato, Z. Xiong, S. Wang, and T. Huang, "AI-generated 6G Internet design: A diffusion model-based learning approach," arXiv preprint arXiv:2303.13869, Mar. 2023.

[302] A. El Saddik, "Digital twins: The convergence of multimedia technologies," IEEE MultiMedia, vol. 25, no. 2, pp. 87-92, Aug. 2018.

[303] A. Clemm, M. T. Vega, H. K. Ravuri, T. Wauters, and F. De Turck, "Toward truly immersive holographic-type communication: Challenges and solutions," IEEE Communications Magazine, vol. 58, no. 1, pp. 93-99, Jan. 2020.

[304] J. Chen, J. Kang, M. Xu, Z. Xiong, D. Niyato, C. Chen, A. Jamalipour, and S. Xie, "Multi-agent deep reinforcement learning for dynamic avatar migration in AIoT-enabled vehicular metaverses with trajectory prediction," arXiv preprint arXiv:2306.14683, Jun. 2023.

[305] J. Chen, J. Kang, M. Xu, Z. Xiong, D. Niyato, and Y. Tong, "Multipleagent deep reinforcement learning for avatar migration in vehicular metaverses," in Companion Proceedings of the ACM Web Conference 2023, Austin, TX, Apr. 2023, pp. 1258-1265.


[^0]:    ${ }^{1}$ The code is available at https://github.com/HongyangDu/GDMOPT

[^1]:    ${ }^{3}$ The website of Amazon Mechanical Turk as a crowdsourcing marketplace: https://www.mturk.com/

    ${ }^{4}$ The website of Datatang: https://www.datatang.ai/

</end of paper 1>


<paper 2>
# Sparks of GPTs in Edge Intelligence for Metaverse: Caching and Inference for Mobile AIGC Services 

Minrui Xu, Dusit Niyato, Fellow, IEEE, Hongliang Zhang, Jiawen Kang, Zehui Xiong,<br>Shiwen Mao, Fellow, IEEE, and Zhu Han, Fellow, IEEE


#### Abstract

Aiming at achieving artificial general intelligence (AGI) for Metaverse, pretrained foundation models (PFMs), e.g., generative pretrained transformers (GPTs), can effectively provide various AI services, such as autonomous driving, digital twins, and AI-generated content (AIGC) for extended reality. With the advantages of low latency and privacy-preserving, serving PFMs of mobile AI services in edge intelligence is a viable solution for caching and executing PFMs on edge servers with limited computing resources and GPU memory. However, PFMs typically consist of billions of parameters that are computation and memory-intensive for edge servers during loading and execution. In this article, we investigate edge PFM serving problems for mobile AIGC services of Metaverse. First, we introduce the fundamentals of PFMs and discuss their characteristic fine-tuning and inference methods in edge intelligence. Then, we propose a novel framework of joint model caching and inference for managing models and allocating resources to satisfy users' requests efficiently. Furthermore, considering the in-context learning ability of PFMs, we propose a new metric to evaluate the freshness and relevance between examples in demonstrations and executing tasks, namely the Age of Context (AoC). Finally, we propose a least context algorithm for managing cached models at edge servers by balancing the tradeoff among latency, energy consumption, and accuracy.


Index Terms-Metaverse, mobile edge networks, artificial intelligence-generated content, generative pretrained transformers, joint caching and inference

## I. INTRODUCTION

Towards artificial general intelligence (AGI) in Metaverse [1], [2], pretrained foundation models (PFMs), e.g., generative pretrained transformers (GPTs) [3], with billions of parameters achieve great success across a variety of fields over the past few years due to their effectiveness at demonstrating emergence abilities in downstream tasks with different data modalities [4]. The pretraining approach offers a reasonable parameter initialization for extensive downstream applications, such as object detection, image generation, and text retrieval. Therefore, PFMs, including language foundation models (LFMs), visual foundation models (VFMs), and multimodal foundation models (MFMs), are in the paradigm of

Minrui Xu and Dusit Niyato are with the School of Computer Science and Engineering, Nanyang Technological University, Singapore 639798, Singapore. Hongliang Zhang is with the School of Electronics, Peking University, Beijing 100871, China. Jiawen Kang is with the School of Automation, Guangdong University of Technology, China. Zehui Xiong is with the Pillar of Information Systems Technology and Design, Singapore University of Technology and Design, Singapore 487372, Singapore. Shiwen Mao is with the Department of Electrical and Computer Engineering, Auburn University, Auburn, AL 36849-5201 USA. Zhu Han is with the Department of Electrical and Computer Engineering, University of Houston, Houston, TX 77004 USA, and also with the Department of Computer Science and Engineering, Kyung Hee University, Seoul 446-701, South Korea. transfer learning that can generalize to new tasks and domains without any task-specific data during pretraining.

PFMs can empower a multitude of intelligent services for Metaverse, such as autonomous driving, digital twins (DTs), and artificial intelligence-generated content (AIGC) for extended reality (XR). For instance, PFMs can facilitate complex driving decisions and generate traffic simulations for autonomous driving [5]. Moreover, PFMs can help understand and respond to human emotions and behaviors during immersive human-avatar interactions. For example, based on the GPT-3 [3], which is an LFM with 175 billion parameters, ChatGPT ${ }^{1}$ enables long and fluent conversations with humans using world knowledge and contextual awareness. In addition to serving PFMs at cloud servers, edge servers equipped with GPU resources can also support fine-tuning and inference processes of Metaverse services, which brings the sparks of GPTs to mobile edge networks. Therefore, deploying PFMs in mobile edge networks allows the provision of low-latency, personalized, customized, and privacy-preserving AI services.

However, compared to cloud servers, resource-constraint edge servers cannot load all PFMs simultaneously to satisfy the requests of services in Metaverse. Aiming at provisioning mobile AI services in edge networks, existing works primarily focus on offloading AI services to cloud servers for remote execution or caching inference outputs at edge servers for low-latency access [6]. On the one hand, offloading PFMs of AI services to cloud servers results in extra core networking latency, traffic overhead, and privacy risks for users utilizing AI services. On the other hand, caching inference outputs at edge servers is no longer efficient for provisioning realtime AI services. Therefore, directly deploying PFMs at edge servers requires effective and fine-grained resource and request management for executing AIGC requests with available computing and energy resources at edge servers.

Specifically, in contrast to existing works on joint service caching and task offloading [7], there are several unique difficulties for joint PFM caching and inference to balance the tradeoff among accuracy, latency, and energy consumption in edge intelligence as follows [8].
- Dynamic Runtime Configuration: During the execution of PFMs, there are varying numbers of requests and performance requirements for downstream tasks, such as accuracy and latency [6].
- Equivalent Model Adaptation: Different PFMs that can be applied to similar downstream tasks in various[^0]

![](https://cdn.mathpix.com/cropped/2024_06_04_88220f7b73a0a0e27aa4g-2.jpg?height=716&width=1791&top_left_y=184&top_left_x=167)

Fig. 1: Categories of PFMs and their characteristic fine-tuning and inference methods. (1)-(3) The workflows of LFMs, VFMs, and MFMs. (a)-(c) The illustration of parameter-efficient fine-tuning. (d) An example of in-context learning.

Metaverse services adaptively [4]. This introduces a challenge for edge servers, as cached PFMs can be used for inference interchangeably to minimize model misses.

- Continuous In-context Learning: PFMs, like GPT-3, can continuously learn and adapt to new domains and tasks based on interactive demonstrations for personalization and customization [9]. The ability of in-context learning enables cached PFMs to improve their performance during inference without parameter updates. This adds complexity in making cache replacement and deployment decisions, as it presents a new tradeoff among inference latency, resource consumption, and accuracy.

To address these issues, this article investigates the potential but scarcely studied problems of PFM caching and inference in mobile edge networks. We first introduce the fundamentals of PFMs for serving mobile AIGC services of Metaverse, and their fine-tuning, and inference methods in edge networks. Then, we present a joint model caching and inference framework in edge networks to serve PFMs of mobile AI services of Metaverse. Furthermore, we discuss potential applications and challenges of serving PFMs for Metaverse services. Finally, to balance the tradeoff among inference latency, resource consumption, and accuracy, we propose a novel metric to indicate the freshness and relevance of examples in demonstrations and current tasks, namely the Age of Context (AoC). The AoC follows the non-increasing utility function that affects the effective examples in context from the entirety of demonstrations resulting from historical interactions. Based on this metric and the number of examples in context, we propose a least context (LC) algorithm to manage PFMs at edge servers. Experimental results demonstrate that the proposed $\mathrm{LC}$ algorithm can reduce the total system cost by improving the accuracy of edgecached PFMs, reducing offloading latency, and utilizing the caching and computing resources of edge servers efficiently.

## II. Serving PFMs of Services in Metaverse

## A. Fundamentals of Pretrained Foundation Models

PFMs belong to the transfer learning paradigm that is used to initialize parameters for downstream tasks. PFMs, such as BERT, GPT-3, Stable Diffusion, CLIP, and ChatGPT, leverage large-scale datasets and pretraining techniques to provide reasonable parameter initialization for various AI services [4]. As shown in Fig. 1. there are primarily three types of PFMs, i.e., LFMs, VFMs, and MFMs, which are widely employed to provide AI services.

1) Language Foundation Models: LFMs, also known as large-scale language models, are PFMs designed to understand, process, and generate human languages. LFMs are trained on massive amounts of text data and can develop a broad understanding of language, including grammar, syntax, semantics, and even some aspects of common knowledge. Two examples of PFMs are GPT and ChatGPT, which have demonstrated impressive abilities in natural language understanding and generation. GPT-3 can enable conversations with humans based on world knowledge and contextual awareness, while ChatGPT is designed to generate human-like responses in a chatbot setting. These models employ self-attention mechanisms to better understand the context and relationships between words in a given text and can be adopted in various downstream tasks, such as sentiment analysis, machine translation, text summarization, question-answering, and text generation.
2) Visual Foundation Models: VFMs specialize in understanding and generating complex images and videos, which are designed to process visual information and generate target outputs. VFMs have shown great potential in advancing the field of computer vision, but they are computing-intensive, particularly during the inference stage. For example, the UNet in Stable Diffusion [10], which is a generative model that can produce high-quality images by iteratively refining a noise vector. Stable Diffusion uses a diffusion process to

![](https://cdn.mathpix.com/cropped/2024_06_04_88220f7b73a0a0e27aa4g-3.jpg?height=440&width=1395&top_left_y=192&top_left_x=365)

Fig. 2: An illustration of the performance of zero-, one-, and few-shot accuracy under different model caching settings [3].

create realistic and high-quality images, and it has been shown to outperform other generative models on a variety of tasks.

3) Multimodal Foundation Models: MFMs can process multiple types of data, such as text, images, and audio simultaneously. They are trained on datasets containing various data modalities to learn the relationships, patterns, and structures within and across different data types. For instance, CLIP is one of the MFMs that classify images based on textual descriptions [11], which uses contrastive learning to train on text and image pairs, distinguishing between positive and negative pairs. During inference, the model takes in an image and a textual description and outputs a score representing the likelihood that the image matches the description, calculated through a dot product. Furthermore, MFMs can be fine-tuned on specific tasks by training them on a smaller dataset.

## B. Fine-Tuning of Pretrained Foundation Models

Fine-tuning refers to the process of improving the performance of PFMs to a specific downstream task by updating its parameters. Since PFMs usually consist of billions of parameters, the fine-tuning process is computationally intensive. Therefore, parameter-efficient fine-tuning of PFMs is utilized for achieving comparable performance to traditional finetuning while reducing resource consumption [12]. As shown in Fig. 1, parameter-efficient fine-tuning can be categorized into three types, including addition-based, specification-based, and reparameterization-based methods as follows [13].

- Addition-based methods involve adding a small number of parameters to the PFMs and fine-tuning them. These methods, which include scalar addition, vector addition, and layer addition, add parameters to the PFMs that are specific to the fine-tuning data. For instance, such parameters include additional layers or heads after the output layer of PFMs.
- Specification-based methods modify the architecture of PFMs to better suit downstream tasks. These methods, such as layer removal, layer replacement, and layer scaling, adjust the PFMs' parameters and architecture to improve performance.
- Reparameterization-based methods reduce the number of tunable parameters in PFMs by reparameterizing their parameters. These methods, such as low-rank factorization, matrix decomposition, and subspace projection, reparameterize the $\mathrm{PFMs}$ to reduce the number of tunable parameters while preserving the PFMs' expressiveness.

Depending on applications such as Metaverse, the finetuning methods can be selected adaptively depending on the resource and performance requirements.

## C. Inference of Pretrained Foundation Models

Different from fine-tuning that updates the parameters of PFMs, the inference is to make predictions on input service requests without changing the parameters. Instead of injecting or updating neural modules in AI models, PFMs can provide accurate output for the task that does not exist in the training, fine-tuning, and inference from instructions and demonstrations from interaction without parameter updates. As shown in Fig. 2, there are three scenarios during the inference of PFMs [3], including zero-shot, one-shot, and fewshot learning. First, zero-shot learning refers to the PFMs that are evaluated on a task for which it has not been explicitly trained. Then, one-shot learning indicates the PFMs need to perform the inference for a new task based on only one example of that task. Finally, few-shot learning implies that a few demonstrations are provided before the inference of the new task. Based on the few-shot learning, the PFMs can perform a meta-gradient in the self-attention layer for adaptation to the new task. Different from fine-tuning, fewshot learning or in-context learning can perform meta-gradient in the attention layers during inference without changing its model parameters. Therefore, few-shot learning can improve the model performance based on examples in instructions and/or demonstrations. However, extra computation consumption and latency are required by processing the examples which depend on the size of the context window in PFMs.

## III. JOINT MODEL CACHING AND INFERENCE

FRAMEWORK

To serve PFMs in edge intelligence for Metaverse, we develop a framework of joint model caching and inference to satisfy service level objectives by utilizing caching, computing, and communication resources in mobile edge networks. Unlike content caching in content delivery networks (CDNs), such as text, images, and videos, the cached models have different cache structures. The cache structure in CDNs is static, with fixed cache sizes and independent of computation
resources [7]. However, due to the flexible configuration of PFMs, the cache structures are dynamic, adjusting to the service requirements in the Metaverse service layer and depending on computation resources during fine-tuning and inference. In the framework, we discuss the model caching configuration and model caching and eviction policy in the PFM layer. Then, we introduce a collaborative mobile edgecloud layer for joint model caching and inference.

## A. Model Caching Configuration

The configuration of each cached PFM consists of the following information.

- Frequency of Use: Frequency of use for PFMs refers to the rate at which a particular model is executed for services in Metaverse. It can be measured in terms of the number of requests per second, the total time spent on processing each request, and other metrics that measure how often a PFM is being utilized.
- Model Sizes: Model size indicates the number of parameters, weights, and other necessary components of PFMs, which affects the latency and energy cost of edge servers for loading and executing PFMs [6].
- Runtime GPU Memory: Runtime GPU memory measures how much RAM or VRAM (video random access memory) is needed by loading the PFM to execute on a given edge/cloud server with its current configuration settings. The runtime GPU memory not only depends on the model sizes but also the runtime precision configuration of the precision. Therefore, there is a trade-off between model precision and GPU memory usage.
- Model Speed: Model speed of PFMs refers to the time complexity or speed at which a particular model can process inference requests. It is usually measured in terms of inference times, i.e., how long it takes for a PFM to complete its task given certain inputs and parameters. Model Speed also has implications on accuracy as faster processing often leads to lower accuracy due to less computation power being available for each request.
- Model Accuracy: Model accuracy of PFMs refers to the degree of correctness or precision with which a model can predict outcomes. For PFMs, model accuracy has implications on speed as higher accuracy often requires more computation power for each request, leading to longer processing times overall.
- Number of Examples in Context: Cached PFMs can accumulate instructions and demonstrations while processing inference requests. The number of examples in context represents the number of related examples in demonstrations the PFMs have gathered. Due to the incontext learning ability of PFMs, the number of examples in context can also impact the accuracy of the models. The size of the context window limits the maximum number of examples in context that can be utilized for each PFM during each inference.


## B. Model Caching and Eviction

Since the cache structure of PFMs is more complicated than traditional web/content caching, the model caching and

![](https://cdn.mathpix.com/cropped/2024_06_04_88220f7b73a0a0e27aa4g-4.jpg?height=651&width=881&top_left_y=192&top_left_x=1077)

Fig. 3: An illustration of the collaborative mobile edge-cloud computing architecture for serving PFMs for Metaverse.

eviction are also more intractable. Model caching and eviction mainly consist of two types of operations, i.e., the passive and active operation as well as binary and partial operation.

- Passive and Active Caching and Eviction: Passive caching is a reactive approach where models are evicted from the cache only when there is not enough GPU memory to load a requested model. Additionally, active caching is a proactive approach where models are evicted and loaded into GPU memory based on predictions of future demand. Active caching can be more efficient than passive caching [6], but requires more sophisticated prediction algorithms and can be less responsive to sudden changes in demand.
- Binary and Partial Caching and Eviction: Binary caching involves loading the entire model into GPU memory before starting inference. In contrast, with partial caching, only a portion of the model is loaded into memory, and inference can begin using that portion. This approach provides a lower level of inference but can be useful when memory resources are limited. When additional memory becomes available, the remaining portions of the model can be loaded into memory, improving inference quality.

If the framework can accurately predict future service request demand, it can leverage active and partial caching to enhance the quality of mobile AI services in Metaverse and reduce resource consumption in mobile edge networks.

## C. Collaborative Mobile Edge-Cloud Caching and Inference

As shown in Fig. 3, collaborative resource allocation among heterogeneous mobile edge-cloud infrastructures is critical in paving the way toward AGI at the edge.

1) Mobile Caching and Inference: Pedestrians and vehicles can process the services to cache and execute PFMs with their local computing resources on mobile devices or devices nearby. This solution can be useful in situations where internet connectivity is limited or unreliable.
2) Edge Caching and Inference: When the local resources of mobile devices and vehicles are not enough for executing PFMs, offloading these services to edge servers via radio access networks become an alternative solution for enabling AI services on edge servers with limited resources. However, due to the limited GPU resources of edge servers, they can only cache several PFMs to react to the user's request. If the edge server does not cache the model requested by the user, it can migrate the user's request to the cloud for execution via core networks or load the model and then execute the model requested by the user. This approach can improve response time and reduce the load on the cloud infrastructure, making it more scalable and cost-effective.
3) Cloud Caching and Inference: Cloud caching and inference solutions involve the utilization of powerful cloud servers to provide almost all PFMs for serving users' requests. However, offloading services to cloud servers incur additional core network latency, which might cause congestion in core networks if there are too many service requests.

## D. Model Caching and Eviction Policy

To design the model caching and eviction policy, three issues should be considered carefully.

- Reducing model miss rate: Actively preloading models and optimizing GPU utilization through dynamic scheduling of AI models can minimize latency and model miss rates, streamlining memory use and request handling.
- Addressing model misses: Handling model misses at edge servers involves offloading service requests to cloud servers, incurring extra core network latency, or loading missing models, leading to switching costs, such as additional latency and energy consumption for allocating resources as well as hardware wear-and-tear.
- Timing model cache decisions: Making cache decisions when the model is first loaded and upon receiving new requests enables dynamic adjustments based on current conditions and usage patterns, promoting efficient caching and lower latency responses.

Therefore, an effective and efficient caching algorithm in this framework should address these three questions properly. Then, we can summarize two remarks for serving PFMs of mobile AI services as follows.

Remark 1: Different from traditional edge content caching in content delivery networks, whose cache structures are static and independent from computation, the cache structures can be dynamic based on the service runtime configuration, such as batch size. This makes the cache loading and eviction of AI services more complex, which requires not only the consideration of user preferences but also the prediction of the intensity of future service requests.

Remark 2: Unlike traditional computation and task offloading in mobile edge networks, where different computation tasks are independent, the inference tasks of PFM-related services are in-contextual. Therefore, before performing these inference tasks, the AIGC model needs to be preloaded into the edge servers' GPU memory, which can cache a limited number of models to provide AI services. Furthermore, as more incontext examples are collected during the interaction, the performance of cached services can be further improved [3].

## IV. Potential ApPlications and ChallenGES

## A. Applications

The potential PFM-based applications in Metaverse include autonomous driving, DTs, semantic communication, and AIGC for XR.

1) Autonomous Driving: Autonomous driving in Metaverse necessitates $\mathrm{AI}$ services such as traffic and driving simulation, which are dependent on computationally-intensive PFMs [5]. To enable this on resource-limited edge servers, model caching, and efficient inference scheduling are essential. In autonomous driving, active model switching can enhance traffic efficiency and safety by adapting to changing road conditions or traffic patterns.
2) Digital Twins: DTs, virtual representations of physical objects or systems, utilize PFMs for AI capabilities like predictive maintenance, anomaly detection, and optimization. The complexity of these systems demands numerous PFMs for accurate modeling across diverse scenarios. Therefore, effective management of the vast number of PFMs, maintaining quality and consistency, and optimizing caching and inference policies can enable DTs to accurately represent physical systems, thus enhancing decision-making and operational efficiency.
3) Semantic Communication: Semantic Communication, a novel paradigm that employs semantic representations, can transform wireless communication systems' design and operation. Its device-to-device pattern enables efficient and secure communication without centralized cloud infrastructure. However, this pattern necessitates advanced model caching algorithms to manage edge servers' limited resources while ensuring cached models' quality and consistency. Implementing progressive caching techniques like active and partial caching can optimize the device-to-device pattern, leading to faster and more reliable AI services on edge servers.
4) AIGC for XR: AIGC is generated by AI methods that utilize PFMs to create content that resembles humanproduced content [12]. To provide AI-generated XR services, multiple PFMs are integrated to handle different types of data and produce relevant and meaningful 3D immersive content. The model caching algorithm ensures that the PFMs work smoothly together, maintaining seamless and immersive experiences for Metaverse users. Achieving this requires careful consideration of the interplay between PFMs and the development of advanced context-aware caching algorithms for efficient cached model management and coordination.

## B. Challenges

1) Dynamic User Service Requests and Objectives: Joint caching and inference services at edge servers face challenges due to dynamic user requests and objectives, such as service latency and accuracy. To tackle these challenges, edge servers must efficiently manage limited resources, ensure cached model quality and consistency, and design joint caching and inference policies to satisfy users' objectives, considering

TABLE I: Detailed parameters and performance of PFMs ( $\mathrm{K}=$ number of examples in context).

|  | Models | Downstream Tasks | Model Size (M) | GFLOPs | $\mathbf{K}$ | Model Performance Score |  |  |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|  |  |  |  |  |  | Zero-shot | One-shot | Few-shot |
| LFMs | GPT3-13B [3] | Translation | 12850 | 26.54 | 64 | 15.45 | 26.12 | 30.83 |
|  |  | Basic arithmetic |  |  | 50 | 3.79 | $15.98 \quad$ | 14.34 |
|  |  | SuperGLUE |  |  | 32 | $54.4 \quad$ | 64.3 | 66.9 |
|  | GPT-3-175B [3] | Translation | 174600 | 354.03 | 64 | 22.03 | 29.63 | 33.77 |
|  |  | Basic arithmetic |  |  | 50 | 25.99 | 40.71 | $49.55 \quad$ |
|  |  | SuperGLUE |  |  | 32 | 58.2 | 68.9 | 73.2 |
| VFMs | UniFormer-S [14] | Image classification | 22 | 3.6 | - | 82.9 | - | - |
|  |  | Video classification | 22 | 167 |  | 82.8 | - | - |
|  |  | Object detection and <br> instance segmentation | 41 | 269 |  | 45.6 | - | - |
|  |  | Semantic segmentation | 25 | 247 |  | 46.6 | - | - |
|  |  | Pose estimation | 25 | 4.7 |  | 74.0 | $\overline{-} \quad$ - $\quad$ - $\quad$ - | $\overline{-} \quad$ - $\quad$ - $\quad$ - |
|  | UniFormer-B [14] | Image classification | 50 | 8.3 | - | 83.9 | - | ![](https://cdn.mathpix.com/cropped/2024_06_04_88220f7b73a0a0e27aa4g-6.jpg?height=43&width=126&top_left_y=723&top_left_x=1791) |
|  |  | Video classification | 22 | 389 |  | 84.0 | - | - |
|  |  | Object detection and <br> instance segmentation | 69 | 399 |  | 47.4 | - | - |
|  |  | Semantic segmentation | 54 | 471 |  | 48.0 | - | - |
|  |  | Pose estimation | 54 | 9.2 |  | 75.0 | - | - |
| MFMs | CLIP-ViT-L/14 11] | Classification | 428 | 175.5 | - | 75.20 | - | ![](https://cdn.mathpix.com/cropped/2024_06_04_88220f7b73a0a0e27aa4g-6.jpg?height=42&width=126&top_left_y=936&top_left_x=1791) |
|  |  | Image retrieval |  |  |  | 71.08 | - | - |
|  |  | Text retrieval |  |  |  | 84.00 | - | - |
|  | CLIP-ViT-H/14 [11] | Classification | 986 | 381.9 | - | 77.97 | $\underline{-} \quad$ - | $\underline{-} \quad$ - |
|  |  | Image retrieval |  |  |  | 73.43 | - | - |
|  |  | Text retrieval |  |  |  | 86.04 | - | - |

factors like model size, frequency of use, and accuracy. Addressing these challenges might require edge servers to develop prediction models for various mobile AI services and propose active caching and inference algorithms.

2) Heterogeneous Model Configuration and Computing Resources: Heterogeneous model configuration and computing resources present challenges in proposing joint model caching and inference algorithms. In detail, PFMs' structure and available edge servers result in varying GPU memory and compute resource requirements, which is typically formulated as an NP-hard mixed integer programming problem, complicating the optimization of caching and inference policies. Moreover, distinct model architectures and computation requirements add complexity. To address these challenges, edge servers must efficiently allocate resources according to each model's specific requirements while considering local computing device availability and capabilities.
3) Context-aware Caching and Inference Algorithms: Codesigning caching and inference algorithms considering contextual information in mobile AI services at edge servers is challenging due to the indirect correlation between model caching and inference duration. Joint policies need to optimize resource allocation according to each model and inference requests' specific requirements while considering user objectives, model size, usage frequency, and accuracy. By codesigning caching and inference algorithms considering the number of examples in context, as shown in Fig. 4, edge servers can utilize extra computation resources for improving the accuracy of PFMs.

## V. Use CaSE of SERVing GPTs in Edge IntelLIGENCE FOR METAVERSE

## A. Mobile AIGC Service Serving Model

We consider an intelligent transportation system in Metaverse system with a remote cloud center, an edge server, and multiple vehicles, serving different Metaverse services, including autonomous driving, DTs, and AIGC-based XR, based on various PFMs. For instance, pedestrians and passengers can immerse themselves in Metaverse with XR by creating and interacting with AI-generated XR content synthesized by PFMs. When users do not have enough resources on their devices and onboard units for executing PFMs, they need to offload requests to edge servers or cloud servers for remote execution. Usually, an AIGC service requires multiple PFMs to work in synergy to satisfy the user's requirements in Metaverse. For example, the Stable Diffusion services consist of three types of PFMs [10], including a Variational Autoencoder that compresses images into a smaller dimensional latent space, a pretrained CLIP ViT-L/14 for conditioning, and a U-Net block that denoises the output from forward diffusion backward to obtain a latent representation.

The detailed parameters and performance of PFMs need to be considered in intelligent transportation systems of Metaverse are listed in Table I, including GPT3-13B [3], GPT3175B [3], Uniformer-S [14], Uniformer-B [14], CLIP-ViTL/14 [11], and CLIP-ViT-H/14 [11]. As we can observe, only LFMs are large enough to have in-context learning ability, while VFMs and MFMs are relatively small. As shown in Table I and Fig. 4, PFMs utilize meta-gradients to learn from context and improve performance as the user interacts with them. Then, the few-shot accuracy can be fit using the data in Table I. Therefore, contextual information has a corresponding

![](https://cdn.mathpix.com/cropped/2024_06_04_88220f7b73a0a0e27aa4g-7.jpg?height=545&width=702&top_left_y=172&top_left_x=251)

Fig. 4: The accuracy in downstream tasks of GPT3-13B/ 175B versus number of examples in context. The few-shot accuracy $a_{2}=a_{0}+a_{1} \log _{2}\left(1+K^{\alpha}\right)$, where $a_{0}$ is zero-shot accuracy, $a_{1}$ is one-shot accuracy, and $\alpha$ is coefficient.

impact on the quality of service provided by AIGC, such as the accuracy of PFMs. Although the introduction of context in PFMs can improve the model performance, the size of the context window also affects the resource consumption and latency during the inference of the model. As shown in Fig. 2, the freshness and relevance of the examples in demonstrations decrease over time until it is no longer pertinent to the current generation task, which is rarely measured in previous work.

## B. Age of Context and Lease Context Algorithm

Therefore, we propose the AoC for evaluating the relevance and freshness of examples in demonstrations that affect the quality of services of PFMs in currently executing downstream tasks. During inference of PFMs, the questions and answers can be recorded in the context windows as examples in demonstrations and instructions. These examples can be leveraged to improve the accuracy of PFMs as they can perform metagradient to fit these examples. However, the meta-gradient might have positive or negative effects on the accuracy, which depends on their quality, relevance, and timeliness. Similar to the age of information (AoI) [15], the AoC indicates the relevance and timeliness of historical contextual examples in demonstrations to the cached PFM and the current inference task. As shown in Fig. 2, the AoC follows the non-increasing age utility function, factoring with a vanishing coefficient of context. Based on the AoC, the number of examples in content can be calculated as the weighted sum of number of examples in demonstrations. Then, the accuracy of PFMs can be obtained by some function w.r.t. number of examples in context as the functions demonstrated in Fig. 4

Finally, we introduce the LC algorithm, based on the AoC, to manage PFMs for mobile AIGC services efficiently. The LC algorithm tracks the number of examples in context, calculating them and removing the cached PFM with the least contexts, i.e., number of examples in context, when GPU memory is needed for loading a new PFM. This approach is effective for large numbers of PFMs on edge servers with limited GPU memory, prioritizing the removal of the least
TABLE II: Detailed system performance comparison for different caching algorithms.

|  | Random | Cloud | FIFO | LFU | LC |
| :---: | :---: | :---: | :---: | :---: | :---: |
| System <br> cost | 25.67 | 7.29 | 27.51 | 5.93 | $\mathbf{4 . 8 8}$ |
| Switching <br> cost | 18.72 | 0 | 23.28 | 0.37 | $\mathbf{0 . 3 2}$ |
| Total <br> accuracy <br> cost | 0.13 | 0 | 0.52 | $\mathbf{0 . 3 6}$ | 0.44 |
| Average <br> accuracy <br> cost | 0.0151 | 0 | 0.0085 | 0.0083 | $\mathbf{0 . 0 0 7 6}$ |
| Inference <br> latency | 0.12 | 0 | 1.30 | 1.32 | $\mathbf{1 . 2 6}$ |
| Offloading <br> latency | 0.04 | 0 | 0.35 | $\mathbf{0 . 2 4}$ | 0.31 |
| Cloud <br> cost | 6.63 | 7.29 | $\mathbf{2 . 0 5}$ | 3.63 | 2.52 |
| Edge <br> Execution <br> Ratio | $9.8 \%$ | $0 \%$ | $\mathbf{7 0 . 7 \%}$ | $49.4 \%$ | $65.0 \%$ |

relevant PFM for the current inference task. Consequently, the accuracy of PFMs of mobile AIGC services is improved by leveraging more contextual information during inference.

In the experiment, we compare the proposed LC algorithm with Random, cloud-only, first-in-first-out (FIFO), and least frequently used (LFU) baselines. With the objective of minimizing service latency and accuracy loss, the system cost is calculated as the sum of the switching cost, the total accuracy cost, the edge inference latency, the edge offloading latency, and the cloud computing cost. As listed in Table II, the performance of the proposed LC algorithm can achieve minimum total system cost while maintaining a high edge execution ratio, which indicates that most of the services are executed at edge servers. Especially, compared with the LFU algorithm, the least context (LC) algorithm can achieve a lower average service accuracy cost by efficiently leveraging the incontext learning ability of PFMs and contextual information.

## VI. CONCLUSIONS

In the article, we have studied edge caching and inference for serving PFMs in edge intelligence for Metaverse. We have proposed a joint model caching and inference framework for bringing the sparks of GPTs to mobile edge networks, toward achieving AGI. Specifically, we have proposed a new metric for evaluating the relevance and freshness of contextual examples and currently executing tasks. Furthermore, we have proposed the LC algorithm for cache replacement to improve the utilization of historical contextual information and thus increase the accuracy of mobile AIGC services. The experimental results demonstrate that the LC algorithm can reduce system costs and improve the execution ratio at edge servers.

## REFERENCES

[1] S. Bubeck, V. Chandrasekaran, R. Eldan, J. Gehrke, E. Horvitz, E. Kamar, P. Lee, Y. T. Lee, Y. Li, S. Lundberg et al., "Sparks of artificial general intelligence: Early experiments with GPT-4," arXiv preprint arXiv:2303.12712, Mar. 2023, [Online]. Available: https://arxiv.org/abs/ 2303.12712

[2] P. Zhou, J. Zhu, Y. Wang, Y. Lu, Z. Wei, H. Shi, Y. Ding, Y. Gao, Q. Huang, Y. Shi et al., "Vetaverse: Technologies, applications, and visions toward the intersection of metaverse, vehicles, and transportation systems," arXiv preprint arXiv:2210.15109, Oct. 2022, [Online]. Available: https://arxiv.org/abs/2210.15109

[3] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell et al., "Language models are few-shot learners," Proc. of the Advances in Neural Information Processing Systems, vol. 33, pp. 1877-1901, Dec. 2020.

[4] C. Zhou, Q. Li, C. Li, J. Yu, Y. Liu, G. Wang, K. Zhang, C. Ji, Q. Yan, L. He et al., "A comprehensive survey on pretrained foundation models: A history from BERT to ChatGPT," arXiv preprint arXiv:2302.09419, Feb. 2023, [Online]. Available: https://arxiv.org/abs/2302.09419

[5] M. Xu, D. Niyato, J. Chen, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, "Generative AI-empowered simulation for autonomous driving in vehicular mixed reality metaverses," arXiv preprint arXiv:2302.08418, Feb. 2023, [Online]. Available: https://arxiv.org/abs/2302.08418

[6] G. R. Gilman, S. S. Ogden, R. J. Walls, and T. Guo, "Challenges and opportunities of dnn model execution caching," in Proc. of the Workshop on Distributed Infrastructures for Deep Learning, Davis, CA, Dec. 2019, pp. 7-12.

[7] J. Xu, L. Chen, and P. Zhou, "Joint service caching and task offloading for mobile edge computing in dense networks," in Prof. of the IEEE INFOCOM, Honolulu, HI, May 2018, pp. 207-215.

[8] Z. Zhou, X. Chen, E. Li, L. Zeng, K. Luo, and J. Zhang, "Edge intelligence: Paving the last mile of artificial intelligence with edge computing," Proc. of the IEEE, vol. 107, no. 8, pp. 1738-1762, Jun. 2019.

[9] Q. Dong, L. Li, D. Dai, C. Zheng, Z. Wu, B. Chang, X. Sun, J. Xu, and Z. Sui, "A survey for in-context learning," arXiv preprint arXiv:2301.00234, Jan. 2023, [Online]. Available: https://arxiv.org/abs/ 2301.00234

[10] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer, "Highresolution image synthesis with latent diffusion models," in Proc. of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, New Orleans, LA, Jun. 2022, pp. 10684-10695.

[11] M. Cherti, R. Beaumont, R. Wightman, M. Wortsman, G. Ilharco, C. Gordon, C. Schuhmann, L. Schmidt, and J. Jitsev, "Reproducible scaling laws for contrastive language-image learning," arXiv preprint arXiv:2212.07143, Dec. 2022, [Online]. Available: https://arxiv.org/abs/ 2212.07143

[12] M. Xu, H. Du, D. Niyato, J. Kang, Z. Xiong, S. Mao, Z. Han, A. Jamalipour, D. I. Kim, V. Leung et al., "Unleashing the power of edge-cloud generative ai in mobile networks: A survey of aigc services," arXiv preprint arXiv:2303.16129, Mar. 2023, [Online]. Available: https://arxiv.org/abs/2303.16129

[13] N. Ding, Y. Qin, G. Yang, F. Wei, Z. Yang, Y. Su, S. Hu, Y. Chen, C.M. Chan, W. Chen et al., "Parameter-efficient fine-tuning of large-scale pre-trained language models," Nature Machine Intelligence, vol. 5, pp. 220--235, Mar. 2023.

[14] K. Li, Y. Wang, J. Zhang, P. Gao, G. Song, Y. Liu, H. Li, and Y. Qiao, "Uniformer: Unifying convolution and self-attention for visual recognition," arXiv preprint arXiv:2201.09450, Jan. 2022, [Online]. Available: https://arxiv.org/abs/2201.09450

[15] X. Chen, C. Wu, T. Chen, Z. Liu, H. Zhang, M. Bennis, H. Liu, and Y. Ji, "Information freshness-aware task offloading in air-ground integrated edge computing systems," IEEE Journal on Selected Areas in Communications, vol. 40, no. 1, pp. 243-258, Nov. 2021.

Minrui Xu (minrui001@e.ntu.edu.sg) received the B.S. degree from Sun Yat-Sen University, Guangzhou, China, in 2021. He is currently working toward the Ph.D. degree in the School of Computer Science and Engineering, Nanyang Technological University, Singapore. His research interests mainly focus on mobile edge computing, deep reinforcement learning, and incentive mechanism design.
Hongliang Zhang [M'19] (hongliang.zhang92@gmail.com) is an assistant professor in the School of Electronics at Peking University, Beiijng, China. He was the recipient of the 2021 IEEE ComSoc Heinrich Hertz Award.

Dusit Niyato [M'09, SM'15, F'17] (dniyato@ ntu.edu.sg) is currently a professor in the School of Computer Science and Engineering, Nanyang Technological University, Singapore. He received the B.Eng. degree from King Mongkuts Institute of Technology Ladkrabang (KMITL), Thailand in 1999 and Ph.D. in electrical and computer engineering from the University of Manitoba, Canada in 2008. His research interests are in the areas of Internet of Things (IoT), machine learning, and incentive mechanism design.

Jiawen Kang [M'18] received the Ph.D. degree from the Guangdong University of Technology, China in 2018. He was a postdoc at Nanyang Technological University, Singapore from 2018 to 2021. He currently is a professor at Guangdong University of Technology, China. His research interests mainly focus on blockchain, security, and privacy protection in wireless communications and networking.

Zehui Xiong [M'20] (zehui_xiong @sutd.edu.sg) is an Assistant Professor at Singapore University of Technology and Design. Prior to that, he was a researcher with Alibaba-NTU Joint Research Institute, Singapore. He received the Ph.D. degree in Computer Science and Engineering at Nanyang Technological University, Singapore. He was a visiting scholar with Princeton University and University of Waterloo. His research interests include wireless communications, network games and economics, blockchain, and edge intelligence.

Shiwen Mao [S'99, M’04, SM'09, F'19] (smao@ieee.org) received his Ph.D. in electrical and computer engineering from Polytechnic University, Brooklyn, NY. He is a Professor and Earle C. Williams Eminent Scholar, and Director of the Wireless Engineering Research and Education Center at Auburn University. His research interests include wireless networks and multimedia communications.

Zhu Han [S'01, M'04, SM'09, F'14] (zhuhan22@ gmail.com) currently is a professor in the Electrical and Computer Engineering Department at the University of Houston, Texas. He has been an AAAS Fellow since 2019. He received the IEEE Kiyo Tomiyasu Award in 2020. He has been a 1 percent highly cited researcher since 2017 according to Web of Science.


[^0]:    ${ }^{1}$ https://openai.com/blog/chatgpt/

</end of paper 2>


<paper 3>
# Industrial Metaverse: Enabling Technologies, Open Problems, and Future Trends 

Shiying Zhang, Jun Li, Senior Member, IEEE, Long Shi, Senior Member, IEEE,<br>Ming Ding, Senior Member, IEEE, Dinh C. Nguyen, Member, IEEE,<br>Wen Chen, Senior Member, IEEE, and Zhu Han Fellow, IEEE


#### Abstract

As an emerging technology that enables seamless integration between the physical and virtual worlds, the Metaverse has great potential to be deployed in the industrial production field with the development of extended reality (XR) and nextgeneration communication networks. This deployment, called the Industrial Metaverse, is used for product design, production operations, industrial quality inspection, and product testing. However, there lacks of in-depth understanding of the enabling technologies associated with the Industrial Metaverse. This encompasses both the precise industrial scenarios targeted by each technology and the potential migration of technologies developed in other domains to the industrial sector. Driven by this issue, in this article, we conduct a comprehensive survey of the stateof-the-art literature on the Industrial Metaverse. Specifically, we first analyze the advantages of the Metaverse for industrial production. Then, we review a collection of key enabling technologies of the Industrial Metaverse, including blockchain (BC), digital twin (DT), 6G, XR, and artificial intelligence (AI), and analyze how these technologies can support different aspects of industrial production. Subsequently, we present numerous formidable challenges encountered within the Industrial Metaverse, including confidentiality and security concerns, resource limitations, and interoperability constraints. Furthermore, we investigate the extant solutions devised to address them. Finally, we briefly outline several open issues and future research directions of the Industrial Metaverse.


Index Terms-Metaverse, industrial, enabling technologies, BC, DT, 6G, XR, AI.

## I. INTRODUCTION

The Metaverse is a nascent concept that presents a model of interconnection between the virtual and real worlds [1]. This digital system integrates both worlds into a cohesive system, providing users with a high degree of freedom to create content and modify modules as they wish, while also allowing different participants to define their own expressions. Some scholars define the Metaverse as a collection of virtual time and space, which leverages technologies such as virtual reality (VR) and augmented reality (AR). People continuously generate new

Shiying Zhang, Jun Li and Long Shi are with the School of Electrical and Optical Engineering, Nanjing University of Science and Technology, Nanjing 210094, China (e-mail: \{shiying.zhang, jun.li\} @njust.edu.cn and slong1007@gmail.com).

Ming Ding is with Data61, CSIRO, Eveleigh, NSW, Australia (e-mail: Ming.Ding@data61.csiro.au).

Dinh C. Nguyen is with the Department of Electrical and Computer Engineering, University of Alabama in Huntsville, USA (e-mail: Dinh.Nguyen@uah.edu).

Wen Chen is with Department of Electronics Engineering, Shanghai Jiao Tong University, Shanghai 200240, China (e-mail: wenchen @ sjtu.edu.cn).

Zhu Han is with the Electrical and Computer Engineering, University of Houston, Houston, Texas, USA (e-mail: hanzhu22 @ gmail.com). content through smart wearables as inputs. It is also considered to be the latest phase of visual immersion technology, which has applications in all areas of production and life as an online digital space. Although there is no precise definition of the term "Metaverse" at this stage, its underlying technical system has matured and has been successfully applied in fields such as education, healthcare, industrial manufacturing, among others. The Industrial Metaverse refers to the establishment of a shared virtual space empowered by Metaverse technologies, to support multi-user, multi-device industrial scenarios for $3 \mathrm{D}$ modeling and immersive interaction.

Especially in the context of the COVID-19 pandemic, which has led to a widespread shift towards remote work, and the continuous advancement of digital technologies, many production activities that traditionally required physical presence can be transformed to enable employees to operate online, similar to a digital twin (DT). However, the difference is that DT is only as a digital replica of the physical elements of the real world, realizing the mapping from the physical world to the digital world. In contrast, Industrial Metaverse is a collective virtual shared space that contains all virtual worlds and derived elements, and virtual space digital individuals for online sharing, interactive perception and content generation. This feature is particularly suited to industrial scenarios, such as detecting part performance through nearly costless fault simulations, creating experimental conditions that cannot be replicated in reality, and expanding operational capabilities through the linkage between the virtual and physical worlds. This can guide physical industrial operations, reduce costs, and improve efficiency by empowering all aspects of the industrial process. Compared to typical virtual worlds, the emphasis of an industrial virtual world lies in simulating realworld industrial processes, manufacturing environments, or operational scenarios. The Industrial Metaverse places a higher demand on cost control for industrial manufacturing, requiring its design and implementation to consider cost efficiency and optimization, aiming to minimize deployment and maintenance costs. Additionally, the Industrial Metaverse involves collaboration among different enterprises in aspects such as industrial production and supply chain management, thus requiring a high level of trust in the execution of interactions. The key benefits of Metaverse are highlighted as follows:

- Low-cost Simulation: The Metaverse can be used to simulate production processes in a cost-effective way. It can be used to implement various services by defining

![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-02.jpg?height=1014&width=1808&top_left_y=173&top_left_x=164)

Fig. 1. The development in the Metaverse from the perspective of technological updates. The concept of the Metaverse can be traced back to the VR concept proposed by Jaron Lanier in 1984. The formal concept of the Metaverse, however, was introduced in 1994. Over the following 30 years, enabling technologies and industries associated with the Metaverse (particularly the gaming industry) have continued to evolve. It was not until 2021 that there was a rapid surge in research achievements related to the Metaverse, which is why 2021 is also referred to as the "Metaverse Year".

virtualized roles and scenarios, to create an orderly hierarchy in an enterprise virtual environment, and to generate content and simulation results that can be used directly to predict and optimize the situation in a real factory.

- Cross-regional Collaboration: The Metaverse transcends geographical constraints and acquires data from diverse sensors and production lines, empowering manufacturers or disparate departments within the same manufacturing entity, geographically scattered, to collaborate on production, thereby enhancing production efficiency.
- Secure Interaction Assurance: Through the integration of security technologies such as Non-Fungible Tokens (NFTs), the Industrial Metaverse can achieve secure protection of digital ownership, thereby aiding enterprises in preventing unauthorized user intrusions and data leaks during collaborative production processes.

In summary, the Industrial Metaverse, as a novel intelligent manufacturing paradigm, is poised to catalyze an unprecedented industrial upgrading, and its inherent value is expected to significantly exceed its consumption value. Fig. 1 shows the development of the main enabling technologies in the Metaverse.

Sure, here's the same table without the blue font color commands:

Here's the table without the blue font color commands:

## A. Related Works

Neal Stephenson's novel "Snow Crash" was the first to introduce the concept of the Metaverse, a digital landscape constructed through AR technology. Since then, there has been an explosion in literature on the Metaverse, with much of it focused on its fundamental characteristics [10], empowering technologies [11], and privacy and security aspects. Some works also analyze the platform deployment and necessary components required for its application scenarios. Our research on the industrial applications of the Metaverse is comprehensive and detailed in Table I

In terms of the empowering technologies within the Industrial Metaverse, the researchers in [6] delineate the technical architecture of blockchain (BC) in Metaverse Industrial 5.0, including its roles and limitations in data collection, storage, sharing, interoperability, and privacy protection. The work of [3] investigates the application of fluid mechanical components in numerical simulation and fault detection, supported by the Metaverse and DT. Additionally, the article by [2] expounds on the motivations and utilities of the four fundamental technologies of the Metaverse: AI, data communication, DT, and mobile edge computing (MEC). Furthermore, it describes seven crucial requirements for the current amalgamation of the Metaverse with the Internet of Things (IoT). Moreover, in [5], the authors present a cross-industry vertical Metaverse reference architecture, elucidating the underlying business logic of the contract interface as the backend, along with the front-end and back-end interaction mechanism implemented

![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-03.jpg?height=536&width=1805&top_left_y=182&top_left_x=152)

Fig. 2. Structure of the survey.

TABLE I

KEY ABBREVIATIONS.

| Abbreviation | Explanation |
| :---: | :---: |
| BC | Blockchain |
| DT | Digital Twin |
| XR | Extended Reality |
| IoT | Internet of Things |
| IIoT | Industrial Internet of Things |
| XRI | XR-IoT |
| VR | Virtual Reality |
| AR | Augmented Reality |
| MR | Mixed Reality |
| MSF | Metaverse Standards Forum |
| SVM | Support Vector Machines |
| LSTM | Long Short-Term Memory |
| GANs | Generative Adversarial Networks |
| PT | Physical Twin |
| FL | Federated Learning |
| VSPs | Artificial Intelligence |
| AI | AI Service Provider |
| ASP | Mobile Edge Computing |
| MEC | Non-Fungible Token |
| NFT | Energy Internet of Things |
| EIoT | Human-computer Interaction |
| HCI | Operations \& Maintenance |
| O\&M | Building Information Modeling |
| BIM | Product Lifecycle |
| PLC | Cyber-Manufacturing Systems |
| CMS | Head-mounted Displays |
| HMDs | Decentralized AI |
| DeAI | Centralized AI |
| CeAI | AIGC Generated Content |
| URLLC | Ultra-reliable and Low-latency Communication |

using the Web3.0 browser engine.

Although some of the works do not specifically focus on industrial applications of the Metaverse, the enabling technologies involved can be directly transferred to the manufacturing industry. For instance, Dionisio et al. [10] provide a comprehensive summary of the four key characteristics of the Metaverse, namely, realism, ubiquity, interoperability, and scalability, which can be leveraged to improve manufacturing operations. Similarly, Wang et al. [12] and Yu et al. [13] analyze the potential of MEC in the Metaverse, and outline the basic components required to deploy MEC in the $6 \mathrm{G}$ network environment, which can be applied in various pre- dictive maintenance scenarios in the manufacturing industry. Furthermore, Kye et al. [14] conduct a thorough analysis of the potential, as well as the constraints, of utilizing the Metaverse in educational contexts. This encompasses the domains of curriculum design and platform development, which could be effectively applied to skill training scenarios within the industry.

## B. Comparison and Our Contributions

We have noticed that the Metaverse shows great potential in the industrial sector and has gradually attracted people's attention. Although there are numerous reviews on enabling technologies such as DT in industrial scenarios, research on Industrial Metaverse is scarce, mostly listing broad application scenarios without exploring the interconnection between enabling technologies or their roles in the architecture of Industrial Metaverse. In the work of [5], although the authors provide Metaverse architecture design for vertical industries, they do not discuss the production processes covered by enabling technologies in industrial production and analyze their specific technical support roles. The work of [15], although addressing certain industry application challenges, lacked comprehensiveness by omitting a discourse on the enabling technologies involved. Overall, at present, other research works have not formed a framework that matches the industrial production scenario, and there is a lack of indepth combing and analysis from the perspective of industrial scenarios and technology applications. Furthermore, existing research lacks a framework that matches industrial production scenarios and lacks in-depth analysis from the perspective of industrial scenarios and technological applications. Additionally, compared to other application scenarios of the Metaverse, the industrial setting necessitates a high degree of real-time responsiveness to ensure synchronization and timely responses with the actual production environment. Any delay in the production environment could result in production line downtime or decreased efficiency. Additionally, interoperability in cross-enterprise collaboration is an indispensable feature of the Industrial Metaverse. Unlike typical commercial Metaverse scenarios, the equipment, systems, and software involved in industrial production usually originate from different vendors,

TABLE II

SURVEY PAPERS OF INDUSTRIAL METAVERSE.

| Ref. | Year | Scope | Technologies | Key Contents |
| :---: | :---: | :---: | :---: | :---: |
| Li et al. 2 | 2022 | Socialization, healthcare, educa- <br> tion, smart city, entertainment, <br> real estate | $\mathrm{AI}, \quad 5 \mathrm{G} / 6 \mathrm{G}, \quad \mathrm{DT}$, <br> $\mathrm{MEC}$ | The authors describe the motivation and utility of <br> using the four Metaverse pillar technologies. |
| Yang et al. $\|3\|$ | 2022 | Fluid machinery pumps and fans | DT, BC, AI, XR | The authors investigate the application of fluid me- <br> chanical components supported by Metaverse and <br> DT in the field of numerical simulation and fault <br> detection. |
| Chang et al. $\|4\|$ | 2022 | Education, product testing, <br> telecommuting, production <br> optimization, smart cities | Computer vision, <br> DT, Network, BC | Three new Metaverse architectures at the edge, and <br> technologies that help the Metaverse interact and <br> share digital data. |
| Bhattacharya et al. $\mid \overline{5}]$ | 2023 | Manufacturing, internet-of- <br> senses, industry, education, <br> vehicle-to-everything, internet-  <br> of-bio-things  | Web 3.0, 6G, BC, <br> NFTs, XR | The authors discuss generalized Metaverse frame- <br> works for modern industrial cyberspace. |
| Mourtzis et al. $\|6\|$ | 2023 | Data acquisition, storage, shar- <br> ing, interoperability | BC, DT | The authors describe the technical architecture com- <br> ponents of BC in Metaverse Industry 5.0. |
| Said et al. $\|\overline{7}\|$ | 2023 | $\mathrm{HCI}$ | $\overline{X R}$ | This work illustrates the complexity of Metaverse <br> online learning from the perspective of HCI. |
| Kshetri et al. $\|8\|$ | 2023 | Economics | $\mathrm{XR}, \mathrm{AI}, 6 \mathrm{G}$ | The global economic impact of the Industrial Meta- <br> verse. |
| Dong et al. $\|9\|$ | 2023 | Task-oriented communications | Semantic commu- <br> nication | Specific requirements for the three main types of <br> tasks in the Industrial Metaverse. |
| Our paper | 2023 | Industry 5.0 (Industrial data col- <br> lection and storage, product de- <br> sign, operation training, system <br> manufacturing, industrial quality <br> control, etc.) | $\mathrm{BC}, \mathrm{DT}, 6 \mathrm{G}, \mathrm{XR}$, <br> $\mathrm{AI}$ | An extensive survey of the Industrial Metaverse. <br> Particularly, <br> - For the first time, we extensively discussed the <br> role of several key enabling technologies at <br> different stages of industry. <br> - We comprehensively analyze the main existing <br> problems and solutions in the Industrial Meta- <br> verse. <br> - We first emphasize more comprehensive open <br> challenges and research directions for the char- <br> acteristics of industrial scenarios. |

utilizing diverse interface standards, communication protocols, and data formats. These elements are intertwined throughout a series of processes including design, production, testing, deployment, and feedback, each involving collaboration among various departments. Lastly, security and privacy concerns are paramount. Industrial production involves a plethora of sensitive data and confidential information such as design blueprints and industrial plans. Therefore, the Industrial Metaverse imposes stricter security requirements and must possess robust security mechanisms to ensure sensitive information is not leaked or maliciously tampered with. Hence, we believe it is necessary to comprehensively analyze and explore the application and deployment of Metaverse technology in industrial settings, and further advance related research endeavors. To this end, this paper provides a set of key contributions as follows:

- Re-evaluate the architecture of the Industrial Metaverse, outlining its current research status and key enabling technologies, while delineating the interactions among these enabling technologies and their roles within the architecture.
- Summarize existing research achievements on Metaverse enabling technologies and their application status in various stages of industrial production, identifying existing deficiencies for each enabling technology.
- Analyze pressing issues in the current stage of the Industrial Metaverse, including privacy concerns, resource allocation, and platform interoperability, while examining existing solutions and shortcomings.
- To advance industrial deployment, we summarize the current standardization efforts and open challenges in the Industrial Metaverse, and provide insights into the future development and research directions of the Industrial Metaverse, offering guidance for future work.

To the best of our knowledge, this is the first comprehensive work to organize and analyze the Industrial Metaverse and its core enabling technologies.

## C. Structure of The Survey

The sections of this paper are structured as follows. Section $I$ presents an overview of the fundamental principles underlying the Metaverse and elucidates the rationale behind its integration into industrial contexts. Additionally, it provides an inventory of pertinent Metaverse literature reviews, comparing them to identify the innovative aspects of our current undertaking. Section II discusses around the five core enabling technologies of the Metaverse and summarizes the industrial applications and existing problems of each technology. Section III succinctly summarizes the four primary challenges encountered in
the Industrial Metaverse. In addition, in Section IV we collate some of the existing standardization efforts and platforms for the Industrial Metaverse. Moving forward, Section V offers a comprehensive compilation of various unresolved hurdles prevailing in the contemporary Industrial Metaverse. Finally, our discourse culminates with a conclusive remark in Section VI Table indicates the list of abbreviations employed in this paper. Fig. 2 outlines the organization of the survey.

## II. The enABLING TEChnOlogIES for METAVerse

This section introduces several core enabling technologies for the Industrial Metaverse, including BC, DT, XR, AI, and $6 \mathrm{G}$. The primary roles of these enabling technologies in the Industrial Metaverse are as follows:

- DT: DT provides virtual representations of physical entities and precise descriptions of products. These fundamental data are provided by industrial sensors and cameras deployed within the enterprise. Combined with AI and autonomous robotics, DT in the Industrial Metaverse is used to represent physical systems in virtual space and meet various constraints and specifications. Simulation, as a critical component of the Metaverse, is used to test vast scenarios to provide optimal modeling and decisionmaking.
- AI: Integrating AI technology can enhance the modeling accuracy of DT. AI can rapidly classify and analyze large amounts of data generated from factories, supply chains, or other production processes, aiding in pattern recognition and decision-making to adjust production processes, improve production quality, and achieve the intelligence and automation of production processes.
- XR: VR and AR serve as immersive tools providing visualization, which are more intuitive, user-friendly, and easier to interpret compared to numbers in tables or points on charts. For example, Rockwell's development team built test scenarios on the Vuforia Studio platform, incorporating CAD files required for training tests to create wiring diagrams mapped to different product layouts and wiring schematics. Leveraging AR technology, Rockwell Automation achieved a 5\% reduction in training time and extended the same training method to other production lines.

If DT, XR, and AI are prerequisites for enterprise decisionmaking and intelligent production in the Industrial Metaverse, then $6 \mathrm{G}$ and $\mathrm{BC}$ provide the necessary low-latency and trustworthy interaction architecture.

- 6G: Metaverse companies use 6G networks to send and receive massive data on their Industrial Metaverse platforms. For instance, when an alert is received at the mission control center, operators can use $6 \mathrm{G}$ connections to remotely view and operate robots via cameras. Researchers at Nokia Bell Labs believe that compared to 5G, $6 \mathrm{G}$ will halve average power consumption and support a 10 -fold increase in peak capacity.
- BC: Lastly, BC, as a trustworthy distributed architecture, effectively enables trusted interactions between multiple enterprise platforms, as well as identity authentication and rights confirmation, which are indispensable in the Industrial Metaverse.

In addition, we have investigated the latest advances in theoretical research and the development of Industrial Metaverse platforms, as well as the application of each enabling technology in different phases of industrial production. In a short summary section for each technology, we briefly summarize some of the issues that still need to be addressed when integrating these technologies with the Metaverse, as well as the technical requirements in industrial scenarios.

Additionally, we have depicted the reference architecture of the Industrial Metaverse in industrial scenarios, as shown in Fig. 3. It includes three layers: the data input layer, the enabling layer, and the industrial application layer. The enabling layer consists of six components: AI, DT, BC, XR, the Metaverse management center, and the data processing system.

Firstly, the data input layer comprises various IoT devices used to collect data sources in the factory, such as cameras, sensors (e.g., acoustic/light/humidity sensors), and user-facing human-machine interaction devices like VR headsets and mobile phones. The collected industrial data includes text data, image data, video data, and other modalities. These multimodal data sources are transmitted to the data processing system for preprocessing, including data cleansing, integration, and transformation. They are stored in the BC through a BC gateway for pre-broadcasting. On the other hand, workers in the Industrial Metaverse can initiate data upload requests themselves. The BC packs and encapsulates the validated data through smart contracts, generating new blocks for continuous updates. When worker submits a data analysis request, they first need to send the request to the administrator. After preliminary authentication by the administrator, the request message is forwarded to the $\mathrm{AI}$ component for data analysis [16]. Meanwhile, the $\mathrm{BC}$ is responsible for recording transactions. The feedback from the data analysis is then used to update the DT. The DT subsequently sends it to the XR layer for $3 \mathrm{D}$ modeling, including modeling of production devices and environmental elements. The decisions obtained are displayed through XR headsets and serve various industrial scenarios in the industrial application layer, including product testing, skills training, and manufacturing. It is worth noting that the data used to update and adjust the DT model can come from the $\mathrm{BC}$ or directly from industrial equipment sensors, internal databases, or manually entered by operators. Through the DT component, the real world can be accurately replicated in digital form on multiple levels of virtual planes and can dynamically interact and synchronize with virtual devices on the virtual plane.

The communication between these layers and components is based on $6 \mathrm{G}$, LoRa and $\mathrm{WiFi}$ communication network to provide the required QoS for the exchanged data. LoRa is a low-power wide area network technology specialized in long range communication for IoT devices. In the Industrial Metaverse, LoRa can support the connection of sensors and IoT devices to the virtual environment. It can be used to monitor and collect data in the real world and transmit this data to the Industrial Metaverse for users to analyze and utilize.

![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-06.jpg?height=1737&width=1637&top_left_y=175&top_left_x=236)

Fig. 3. Industrial Metaverse Architecture. It consists of three layers: the data input layer, the enabling layer, and the industrial application layer. The enabling layer includes six components: AI, DT, BC, XR, the Metaverse management center, and the data processing system. In this context, the data input layer is composed of various industrial IoT devices. The collected multimodal industrial data is transmitted to the data processing system for preprocessing. The processed data can be stored in the $\mathrm{BC}$ or streamed to the $\mathrm{AI}$ component for analysis. The results of data analysis are used to respond to requests from the Metaverse management center, and decision outcomes are fed back to the BC. The BC performs updates on the network and DT. The DT achieves proportional replication and synchronization of virtual and physical devices and transfers parameter information to XR for 3D modeling. XR displays decision solutions through human-machine interaction devices and is applied in various industrial scenarios.

In the ensuing discourse, we shall expound upon each of these constituent strata comprising the foundational edifice.

## A. Blockchain

In the realm of the Industrial Metaverse, nodes of the Industrial Internet of Things (IIoT) amass vast quantities of sensor data in order to achieve instantaneous correspondence between the virtual and physical domains. Nevertheless, this pursuit carries with it the hazard of security and privacy breaches, as the information of these nodes is profoundly susceptible to compromise [16]. Moreover, there exists a necessity to accomplish cross-platform interoperability within the Metaverse, grounded in a specific framework.

$\mathrm{BC}$, as the fundamental technology of the Metaverse, is well-suited to meet the aforementioned requirements. Firstly, it leverages identity verification and consensus mechanisms to

![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-07.jpg?height=350&width=879&top_left_y=188&top_left_x=165)

Fig. 4. The vulnerabilities, advantages, and application stage of $\mathrm{BC}$ in Industrial Metaverse.

ensure the privacy and security of users, as well as the integrity of vast amounts of industrial data, while also providing a comprehensive transaction audit trail. Secondly, due to the decentralized nature of $\mathrm{BC}$, it enables collaborative production among multiple manufacturers, allowing managers to schedule and coordinate activities across multiple platforms without the need for third-party verification platforms [17]. Fig. 4 shows the vulnerabilities, advantages, and application stage of $\mathrm{BC}$ in Industrial Metaverse.

1) Research Advances of Blockchain Enabling Industrial Metaverse: Initially, the question of safeguarding privacy in the Metaverse was explored in the financial sector, where individuals use NFTs as digital property markers to attain secure data correspondence. Each NFT is linked to an owner, and ownership records are retained on a server. Nonetheless, due to the more intricate real-time information interaction and the heavy workload associated with production tokens in the context of industrial manufacturing scenarios, it has become necessary to establish a collaborative governance framework among enterprises that is composed of more comprehensive components and architecture. To tackle this predicament, Lin et al. [18] propose a value-oriented, trusted, collaborative governance architecture as a blueprint for the deployment of the Metaverse in intelligent manufacturing scenarios. The authors provide a thorough summary of the functions and benefits of $\mathrm{BC}$ at various system levels, such as trustworthy data storage, secure transactions, and regulation.

It is noteworthy that computing resources, functioning as integrated asset exchanges, are also included in the domain of the Industrial Metaverse. However, the aforementioned architecture does not account for the issue of energy constraints caused by IIoT nodes. When devising an integration method between the Industrial Metaverse and BC, resource optimization and sharing should be fully taken into account. For instance, an enterprise management platform can employ various communication methods to offer computing power and encapsulate the communication steps of factory-deployed devices into a protocol. In this way, less powerful computing devices can utilize the support of stronger computing power.

The work presented in [19] introduces a communication paradigm for the native Metaverse, where users can access using encrypted addresses. A dedicated frequency band in the regional spectrum is utilized for the transmission of BC links in public domains. The controller can categorize access to specific resources to eliminate integration barriers between the
Metaverse and inter-chain links. However, this encrypted address does not provide precise network topology information, and industrial IoT devices may possess mobility features, leading to a constantly changing topology structure that makes it impossible to determine the connection mode between nodes. Another practical concern is that profit-driven manufacturers typically consider whether the additional benefits of industrial synergy can cover participation costs. A potential solution is to design incentive mechanisms to attract companies to participate in routing-related activities. To address these two issues, the authors of [20] propose a layered BC architecture with a main chain and multiple sub-chains for sensor data analysis, combined with federated learning (FL) and crosschain technology. Additionally, an age-based smart contract is utilized as an incentive mechanism to achieve trustworthy model aggregation and updates. Regarding the architecture design for dynamic topology scenarios, the authors of [21] adopt another approach for the architecture design, based on MEC. This solution allows MEC servers to possess additional resources for resource sharing and optimized utilization in non-trusted environments. This framework takes into account the heterogeneity of available server resources and user task requests, and utilizes a task assignment mechanism for edge resource sharing. This BC-based MEC deployment scheme can significantly enhance the utilization of computing resources while also ensuring system security.

Numerous studies have examined distinct scenarios for amalgamating the Metaverse with BC, such as architectural modeling and remote healthcare. Though these scenarios may not be relevant to our current discussion, they may potentially be customized to the industrial production domain, such as in the modeling of production devices and remote equipment control. For instance, Building Information Modeling (BIM) technology, which is utilized in 3D architectural modeling, can be exploited through parameterization methods to facilitate speedy data computation and storage for project management processes. After exploring the components of BIM, the application of the Metaverse in virtual world construction, and the latest $\mathrm{BC}$ research, the authors of [22] put forward a preliminary BC-based BIM data trading framework. Moreover, the author also accentuates several potential future research topics, such as immersive design experiences in the Metaverse, the evolution, and retrospection of architectural scenarios. Bhattacharya et al. [23] raise the issue of security attacks in remote surgery, which can result in the uncontrollability of the mobile robot arm. Gupta et al. [24] initially employ the hyper-ledger fabric channel in remote surgery. In subsequent work, the author further established a BC-based collaborative access control system, which comprises three parts: the master domain, the Metaverse control, and the subordinate domain, and all metadata is encapsulated in the ledger. Thus, the operator's behavior in the Metaverse can be recorded and utilized to enhance the robot's behavior feedback.

In general, the exploration of synergizing the Metaverse with $\mathrm{BC}$ is still in its nascent phase, with applications primarily focused on digital asset management, smart contracts, and decentralized identity verification to enhance the security and sustainability of the Metaverse.

2) Application Stage in Industrial Production: The integration of $\mathrm{BC}$ in industrial applications can bring forth a multitude of benefits. One such application is in the domain of supply chain and logistics, where $\mathrm{BC}$ can provide a solution to the lack of traceability and transparency in traditional supply chain management. By enabling the tracking of the entire life cycle of a product, from raw materials to finished goods, BC can ensure safe and convenient management of commodity flow [25].

In industrial data collection, the non-repeatability of data is crucial for accurate data analysis and effective business decisions. With BC, the hash value of each data block can be calculated and compared with the previous block's hash value to ensure that the collected data is not duplicated, thereby improving the reliability of data collection.

In terms of data storage, the massive amount of multimodal data generated during industrial production processes can quickly exhaust physical storage capabilities [6]. Overreliance on centralized storage systems in the Metaverse can result in a single point of failure, leading to significant economic loss. BC's decentralized paradigm can effectively solve the problem of single point of failure while preventing data tampering through the continuous creation of new blocks.

Furthermore, BC can promote cross-enterprise production collaboration and platform interoperability by realizing crosschain protocols and facilitating cross-chain transactions and value flow across different devices [26]. At the enterprise level, it can enable collaboration among suppliers, manufacturers, and end-users, as well as cross-chain governance among different enterprises during the production stage.

3) Summary: BC can provide efficient management for the Industrial Metaverse platform through mechanisms such as non-tampering, traceability, collective maintenance, and openness and transparency [27]. However, in a BC network, each participating node must verify transactions, which can lead to latency issues due to network transmission, disk writing, and other operations. Although several studies have been proposed to optimize the network topology, design new consensus algorithms, and adopt a hierarchical structure to alleviate this problem, these solutions are limited by the size of the network and the number of nodes. Therefore, balancing decentralization, security, and scalability under the Industrial Metaverse architecture is an urgent problem that needs to be addressed for sustainable enterprise development.

## B. Digital Twin

DT can produce a virtual replica of a physical factory by gathering and analyzing production data and equipment parameters in the actual facility. With this information, DT can perform behavior predictions of physical objects or machine states. By accurately mirroring the real world in the virtual world, DT can resolve complex issues that would otherwise prove challenging to solve.

Hence, to develop an Industrial Metaverse platform, DT must fulfill three vital requisites: the capability to simulate industrial entities, the capacity to dynamically procure data from the live environment, and the proficiency to synchronize the data in real-time. Fig. 5 shows the vulnerabilities, advantages, and application stage of DT in Industrial Metaverse.

1) Research Advances of DT Enabling Industrial Metaverse: DT made its debut in the industrial sphere in 2003, when Michael Grieves proposed its implementation in product lifecycle (PLC) management. By utilizing production data and digital models, simulated processes became possible. As the IIoT and DT evolved, the primary focus shifted towards using DT to reduce industrial sites and enhance product evaluation through assistive technologies.

However, some researchers argue that DT tends to overlook crucial action details and treat operators and small devices as mass points in simulations, affecting object modeling. To overcome this challenge, Zhou et al. [28] design a small object detection framework that fuses multidimensional sensor data, treating devices, products, and operators as fundamental environmental parameters. They implemented multi-level feature fusion based on hybrid neural networks created with MobileNetv2, YOLOv4, and Openpose. To effectively reduce the false detection rate, Wu et al. [29] utilize multi-modal fusion data as a data source, Cyber-Manufacturing Systems (CMS) interface for numerical insertion and visualization in DT for product surface defect detection tasks. Finally, they employed morphological alignment algorithms. Numerous studies have revealed that improving the simulation capability of DT requires integrating and optimizing $\mathrm{AI}$ algorithms.

Furthermore, given that DT necessitates substantial bandwidth and computing resources, it inherently demands realtime synchronization with stringent reliability and low latency prerequisites [30]. As a result, achieving real-time synchronization for DT entails stringent reliability and low latency requirements. Therefore, an important research topic is the deep integration of DT into industrial platforms. This includes designing a shared architecture for digital factories and developing scheduling algorithms to optimize resources. A novel approach is to utilize DT in conjunction with edge intelligence to facilitate immersive Metaverse connectivity. By optimizing resource allocation, ultra-reliable and low-latency communication (URLLC) can be realized [31]. The authors in [32] explore the bi-directional dependency of DT with MEC and the interaction of DT-assisted edge computing with DT to avoid Metaverse service disruptions. Dai et al. [33] integrate lightweight DT with MEC to reduce cloud load, increase the number of service requests for MEC, and significantly reduce the time complexity of the algorithm by introducing Merkle trees. Although there have been numerous studies on industrial DTs for handling edge selection and task offloading, these schemes often overlook the data storage limitations of the Metaverse platform. In fact, the fundamental components of the Industrial Metaverse can be simply divided into virtual service providers (VSPs) and individual users and enterprises. However, not all physical assets can be deployed on the platform to capture device status information in real-time to ensure full interoperability. Han et al. [34] propose a dynamic layering of platforms through IoT devices, where VSPs maintain business profitability by increasing synchronization strength. The IoT devices are responsible for transmitting data to independent VSPs. The proposed architecture achieves

![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-09.jpg?height=374&width=875&top_left_y=190&top_left_x=164)

Fig. 5. The vulnerabilities, advantages, and application stage of DT in Industrial Metaverse.

optimal synchronization through optimal control theory. Zhang et al. [35] investigate the problem of multi-base station channel resource allocation in a Metaverse DT, using an endto-end cloud three-layer framework. They design a reverse auction mechanism to solve the resource allocation and pricing problem. Furthermore, the authors in [36] consider the joint design of communication, computation, and storage resources to formulate the problem of minimizing latency by optimizing edge task offloading and arithmetic overhead. This improves the Quality of Experience of DT in the Metaverse.

DT enables the digitization of physical objects, while XR technology provides the medium for interaction between the two. Therefore, in addition to DT itself, a part of the works investigate the synergy between DT and XR technologies in industrial scenarios. For instance, Yang et al. [37] propose a framework for smart crane applications based on DT and XR to improve program development efficiency and Humancomputer Interaction (HCI) system usability. Coupry et al. [38] investigate options for XR technologies to improve BIM and highlight the challenges of implementing BIM-based systems. Jerov et al. [39] investigate the prospects of DT and XR applications in control systems, where they develop a DT system for XR with basic interaction mechanisms and connect it to real production scenarios. While some of the above works are limited to conceptual design and specific use cases, a collaborative approach is needed to build a system that integrates DT and XR effectively.

Tu et al. [40] propose a TwinXR approach that effectively merges information management DT with knowledge-based XR technology. By leveraging IoT devices available in the factory, XR programs can be swiftly developed and instantiated, while the development process is greatly facilitated through the use of publicly available software packages. The authors effectively demonstrate the remarkable compatibility and versatility of this approach by showcasing its successful application to cranes and robot arms in smart factories. Nonetheless, the authors duly acknowledge that the adoption of TwinXR methods can be prohibitively expensive when the machinery is stationed at a specific, static position, rendering cost a critical limiting factor in real production settings.

It is noteworthy to mention that, in addition to the collaborative architecture research, the real-time simulation characteristic of DT allows for monitoring the dynamic properties of physical portals for detecting various network attacks. Gupta et al. [41] suggest synchronizing the device-generated traffic through the gateway router, utilizing Support Vector Machines (SVM) for achieving regular synchronization with DT. This framework attains an accuracy rate of over $93 \%$ in detecting malicious traffic.

2) Application Stage in Industrial Production: Product design stage: During the product conception stage, companies utilize DT to test the production environment, constructing a virtual model to improve production by adjusting parameters in order to reduce design defect rates and production costs, while meeting preset requirements. This process entails continuous iterative optimization of both digital and physical models, interconnecting through data flow [42].

Supply chain and Operations \& Maintenance (O\&M): DT can optimize the production process in remote equipment O\&M and supply chain management. In equipment operation and maintenance, DT provides real-time data on equipment performance, including its current condition, remaining useful life, and any potential problems or failures that may occur. In the supply chain management segment, DT creates a digital copy of the supply chain network, enabling managers to monitor and optimize their operations in real-time.

Post-commercialization services: Furthermore, the Industrial Metaverse platform integrated with DT can be used to manage the traditional industrial service phase, providing DT information on the product after it has been finished and put into the market. Both manufacturers and technical service providers can access product status information on the cloud platform, which can be used to help engineers and designers improve customer experience through customized services.

3) Summary: To put it succinctly, DT enables the integration of physical and digital worlds through real-time interaction and data analysis, providing a new perspective for addressing industrial challenges [43]. By combining explicit and tacit knowledge, real-time operational data can be incorporated into traditional industrial models to build DT that aid in industrial management. However, further research is needed to consider constraints within the Metaverse [44], such as security and storage, as well as to explore integration with other enabling technologies, as current efforts in this area are still in their infancy.

## C. $6 G$ Communication Network

The Industrial Metaverse is typically data-driven and sensitive to perception latency at the physical layer. However, sensor data at the physical layer in industrial scenarios is often complex, with characteristics such as high volume, multiscale, and high noise. Additionally, the supporting technologies involved in the Industrial Metaverse often impose high demands on communication bandwidth. For example, NFTs consume significant bandwidth during digital asset validation, and the verification of digital tag sets in $\mathrm{BC}$, as well as the bandwidth requirements for transactions involving different entities [45]. Furthermore, VR systems require a bit rate of $1 \mathrm{Gbps}$ to provide qualified video streams visually, and real-time feedback for remote operations also necessitates high reliability and high data connection rates.

5G's URLLC are both unable to meet the aforementioned requirements. The next-generation wireless communication
technology represented by $6 \mathrm{G}$ can provide high throughput and low-latency guarantees for the interaction between the physical and virtual worlds' upper and lower-layer links. This is achieved through fast transaction processing to achieve efficient management. In addition to throughput, the goal of the Industrial Metaverse is to ensure high concurrency of devices in space and time, allowing all devices and users on the platform to access and leave anytime, anywhere. This increases the demand for seamless and comprehensive connectivity provided by $6 \mathrm{G}$. In this regard, $6 \mathrm{G}$ can provide auxiliary connections through coordination with IoT devices and achieve global coverage, while its intrinsic security technology can be used to defend against unknown security threats. Fig. 6 shows the vulnerabilities, advantages, and application stage of $6 \mathrm{G}$ in Industrial Metaverse.

1) Research Advances of $6 G$ Enabling Industrial Metaverse: Most existing works are optimized for 6G networks and integration with the Metaverse, and metric focus varies for different application scenarios. The performance metrics in the Industrial Metaverse include throughput, latency, reliability, Age of Information, and energy efficiency [46], among which all metrics' weights, except latency and reliability, are deeply connected and affect joint architecture design. In this regard, Cao et al. [47] discuss several metrics' coupling relationship in-depth and argues that communication delay, reliability, and AoI are the three most crucial metrics in an IIoT environment with heterogeneous traffic. The authors collect data and control heterogeneous traffic from the short packet transmission optimization perspective. However, their specific task division is not given, and transmission jitter's performance loss is not negligible in the actual industrial environment.

Another approach is based on overall architecture design. MEC, an essential auxiliary mechanism of the 6G network, can significantly reduce request delay due to its proximity to the source. The devices in the factory, following the edge computing hierarchy, can include sensors, IIoT devices, and servers [12]. Sensors collect industrial data, and various IIoT device states are responsible for receiving and then converge to the cloud for unified processing. In a typical scenario, some lowerlevel operations in industrial production can be decentralized to the lightweight mobile end, and the administrator can control the device remotely. Various $6 \mathrm{G}$ mobile edge technologies exist for application to the Metaverse, including wireless communication technologies (heterogeneous radio, intelligent reflective surfaces, non-orthogonal multiple access, etc.) and new communication technologies (semantic communication, holographic-type communication, tactile communication, etc.) [13].

Aslam et al. [48] present a layered Metaverse design from the perspective of layers, with different layers connected by a data exchange layer. They then demonstrate the autonomous driving remote control case and finally discuss the challenges and propose solutions for $6 \mathrm{G}$ and Metaverse integration, including how to deal with wireless communication security threats, submillimeter beamforming loss, and channel interference.

Since 6G mobile authorized devices in the same production environment need to share computational memory and

![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-10.jpg?height=369&width=878&top_left_y=176&top_left_x=1076)

Fig. 6. The vulnerabilities, advantages, and application stage of $6 \mathrm{G}$ in Industrial Metaverse.

frequencies, unreasonable resource allocation can easily lead to network congestion, which we will cover in detail in the next section.

2) Application Stage in Industrial Production: Computer Vision Scenarios with Multiple Accesses: With the advent of 6G-enabled industrial IoT devices, computer vision scenarios can be significantly enhanced with seamless connections and high data transfer rates, enabling better image quality in the Metaverse. Additionally, MEC resources in a virtual $6 \mathrm{G}$ network can leverage deep learning techniques to process heterogeneous devices in the enterprise [45].

Product Decision and Service Phase: The product decision and service phase can also benefit from $6 \mathrm{G}$, as it facilitates computation-centric communication by exchanging communication resources to ensure the required quality of service for adequate computational accuracy. This, in turn, improves decision-making efficiency and enables further automation. Moreover, 6G can simplify the supply chain network, leading to agile and adaptive service provisioning for Metaverse applications.

3) Summary: The network heterogeneity of each Metaverse subplatform of manufacturers presents a significant challenge for the development of a unified communication network, requiring a common network access and control protocol. Additionally, plants that deploy $6 \mathrm{G}$ devices must ensure the compatibility and operability of existing network-enabled devices, which must be fine-tuned to each individual subsystem. Currently, there is a lack of available 6G-enabled IIoT architectures for deployment on Industrial Metaverse platforms, as well as a lack of corresponding standards and specifications. Although relevant reference architectures have been proposed [49], the hierarchical relationships are oversimplified and require further empirical testing, and the theoretical aspects remain unexplored.

## D. Extended Reality

In manufacturing scenarios, there are instances wherein it becomes imperative to employ diminutive sensors or IoT devices at the distant terminus, which typically lack interfaces conducive to Metaverse user engagement. XR, including VR, $\mathrm{AR}$, and mixed reality (MR), serves as the interface between the tangible realm and the ethereal domain of the Metaverse, endowing users with a myriad of multifaceted interaction modalities [50]. Within this realm, VR unveils itself as a
veritable medium for sensorial emulation, employing headmounted apparatuses to engender immersive experiences. It facilitates the visualization of diverse design iterations, verifies procedural efficacy, and bestows captivating verisimilitude. On the other hand, AR seamlessly integrates virtual components, such as textual and pictorial elements, into the fabric of reality through handheld devices like smartphones and tablets. By availing themselves of this technology, operators are endowed with instantaneous access to real-time equipment status updates, virtual cues, and instructional guidelines. Furthermore, MR transcends conventional overlays of virtual content upon tangible surroundings, instead forging a harmonious convergence between the virtual and the corporeal, facilitated by perceptive and interpretative faculties attuned to the veritable realm.

In broad terms, constrained by diverse factors in industrial scenarios, comparatively substantial apparatuses like robotic arms, for instance, manifest greater aptitude for remote manipulation by technicians, whereas XR can function as an ad hoc controller. Furthermore, XR technology has the capability to furnish a fusion of tangible and virtual domains for usermachine interplay, thereby lending itself to deployment within industrial production scenarios for the purposes of operational guidance, equipment maintenance training, and product modeling and design. Fig. 7 shows the vulnerabilities, advantages, and application stage of XR in Industrial Metaverse.

1) Research Advances of XR Enabling Industrial Metaverse: In a certain context, XR constructs an adaptable and fluid realm for visualizing virtual content within the realm of MR. However, a primary concern arises regarding the imperative of upholding the sustainable connectivity attributes of content within the physical realm, as virtual contrivances in the Industrial Metaverse persistently expand. Should an information disjunction arise between the Metaverse and the physical space, coupled with a dearth of consistent communication, it can potentially engender cognitive overload for the user [51]. Put simply, when operators engage in interactions within the virtual plant, they are typically unable to access and manipulate the genuine physical apparatus. Consequently, it is crucial to address these disconnections within the Metaverse in a scientifically comprehensive manner, as failure to do so can easily impinge upon critical industrial scenarios.

An intriguing concept in this realm is the utilization of an extended Metaverse agent approach based on the MiRAs cube theory [52]. This approach serves the purpose of dynamically governing virtual device control and hologram modeling, while facilitating interactions with other holograms and operators to manifest within the physical realm. Building upon this notion, Rincon et al. [53] leverage agents founded on the meta-model (MAM5) and JaCalIVE frameworks to mitigate discrepancies between simulated and real systems. Intelligent resource artifacts are incorporated, enabling seamless access to the physical world for devices. The dynamic essence of agents is further explored by Croatti et al. [54], whose architecture allows for agents to be dynamically altered at runtime, employing diverse heterogeneous agent programming techniques to modify agents within AR. In addition to system modeling, certain researchers have embraced agent-based approaches to automate XR quality testing and support experience evaluation by adopting user roles [55]. Braud et al. [56] construct an XR architecture from the operating system level, which perceives the physical world as a shared resource among applications. By implementing meticulous control mechanisms through several components such as environment understanding, specialized chip drivers, networking, interoperability, and various other interconnected elements, they achieve enhanced integration of content with the physical-digital world, thereby attaining superior performance. Drawing upon the extended Metaverse architecture, Guan et al. [51] conduct an in-depth analysis of interface design consistency within the Metaverse. They attribute inconsistencies to signal noise and propose the utilization of "noise reduction" techniques to mitigate disparities between the Metaverse and the physical world. This approach aids in facilitating the development of interoperable interfaces. The authors further delve into the realm of seamless integration by exploring the presence of interoperable agents. They showcase the effectiveness of this approach through two early design projects, presenting interactive and tangible Metaverse applications that can be deployed across both physical and virtual realities. As interfaces of this nature continue to be researched and developed, the integration of suitable filters and controls becomes crucial in ensuring the sustained continuity of the Extended Metaverse. This integration holds the potential to serve as a valuable research direction in the long-term future.

Another intriguing concept in interactive device interfaces is the utilization of a convergent technological paradigm known as XR-IoT (XRI) systems, which combine XR and IoT. These XRI systems capitalize on the complementary attributes of both domains, leveraging the interconnectivity of environmental entities facilitated by IoT, while XR empowers informationrich interactive interfaces and serves as a platform for $\mathrm{HCI}$ [57]. The evaluation of interface design in XRI systems is initially proposed by Dnser et al. [58], who delineat the original design intent of XRI and presented a comprehensive evaluation framework from the perspective of interface design, with usability at its core. Building upon this work, Tsang et al. [59] provide a more exhaustive taxonomic approach and a QoE reference standard for the development and assessment of XRI systems. In the context of real-world industrial scenarios, Morris et al. [60] envision an immersive, multi-user, and multiagent-driven XRI-based intelligent workstation system. This system incorporates haptic feedback-equipped desks, cameras, and other devices to monitor personnel status, while physical laptops run machine learning models to determine personnel conditions. Taking it a step further, Oppermann et al. [61] introduce a $5 \mathrm{G}$ XR toolbox tailored for remote maintenance scenarios in industrial equipment. This toolbox facilitates MR and VR views by establishing digital connections between field equipment and personnel. Operating on actual machines within real production lines, the system utilizes authentic CAD data to create a realistic prototype environment.

Another intriguing perspective to explore is whether the utilization of AR/VR devices can have adverse effects on workload, considering that workload predominantly reflects the input costs in production tasks. Kantowitz et al. [62]

![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-12.jpg?height=384&width=875&top_left_y=182&top_left_x=164)

Fig. 7. The vulnerabilities, advantages, and application stage of $\mathrm{XR}$ in Industrial Metaverse.

demonstrate that efficiency tends to decline as task complexity increases during HCI, even within the maximum workload capacity. To put it simply, it raises the question of whether operators need to exert additional cognitive effort to execute tasks through VR interfaces that could otherwise be performed in the real world. In this context, Xi et al. [63] conduct an investigation on the workload disparities of XR across six distinct dimensions. Their finely detailed analysis reveal that AR has a more pronounced impact on workload compared to VR, while the combination of both technologies do not escalate task difficulty. Future research endeavors can delve into exploring the mechanisms and boundary conditions of these different dimensions, enabling XR designers to develop more scientifically informed HCI systems.

2) Application Stage in Industrial Production: Product Recognition and Modeling: Object detection is widely applied in XR and is an essential task for achieving the Metaverse. For example, computer vision techniques are used for real-time recognition and overlaying relevant information on the display screen, such as product specifications, usage instructions, and maintenance methods. It can also be applied to the modeling and visualization of industrial products, where real-time visual feedback helps improve the efficiency of product design, assembly, and maintenance.

Device Pose Recognition: In XR, it is often necessary to observe and recognize the actions of devices like robots in a 3D immersive factory environment and provide feedback for specific actions. For instance, on an assembly line, AR technology can be used to detect the pose of devices or components in real time, including rotation angles, positions, and sizes, thus assisting operators in correct assembly and adjustment.

3) Summary: The Industrial Metaverse is a collaborative virtual space that enables enterprise employees to interact and work together in a digital environment. By manifesting as tangible virtual images within this space, operators can engage in virtual production and collaboration that runs parallel to the real world, thereby enabling digital transformation. This immersive technology is poised to shape the new form of the Industrial Internet. Additionally, AR and MR technologies have the potential to revolutionize the operation of the physical world within factories. As digital entities transition from a purely virtual environment to a physical one, further industrial design and technological considerations are necessary to ensure the success of digital transformation.
Here's the table without the blue font color commands:

## E. Artificial Intelligence

As the foundational cornerstone of the Industrial Metaverse, AI obtains its input data through an array of sensor devices situated within the factory. This data, once subjected to meticulous preprocessing procedures, serves a multitude of industrial production processes, encompassing the realms of image and audio recognition, astute decision-making support, and production planning, among others [73]. Concurrently, AI remains attuned to user interactions transpiring within the virtual realm. Furthermore, beyond its inherent industrial applicative worth, AI possesses the potential to harmoniously unite with other pivotal enabling technologies, thereby bestowing an enhanced HCI experience. All in all, the pivotal role of AI technology lies in its provision of essential implementation reliability and exceptional performance standards to propel the Metaverse forward.

1) Research Advances of Artificial Intelligence Enabling Industrial Metaverse: At this juncture, the research endeavors surrounding AI in the Industrial Metaverse converge upon two primary focal points. Firstly, the synergy between AI and other enabling technologies takes center stage, encompassing algorithm optimization for VR devices, as well as deployment paradigms stemming from the amalgamation of $\mathrm{BC}$ and MEC. Such combinations give rise to distributed AI and edge intelligence, which find practical application within the Industrial Metaverse. Secondly, dedicated technical research pertaining to industrial AI itself unfolds, including domains such as natural language processing, speech recognition, and the notable emergence of generative AI models, typified by chatGPT, which exhibit broad applicability in domains like intelligent $\mathrm{Q} \& \mathrm{~A}$ and assisted decision-making.

First, under AI-assisted mechanisms, XR includes technologies such as computer vision and XR computing, which can immediately capture environmental data within the field of view and convey real-time processing results to users through display devices. On one hand, research has been conducted on adaptive architectures for XR devices. Wong et al. [69] devise a system called GetWild, where the main network in the AI stage is CGAN. This stage allows users to capture images using VR devices as input to create rough 3D objects and terrain, and further perform editing operations. This technology can be effectively applied in the industrial product design process. However, this massive 3D modeling can pose resource constraints for small edge devices such as VR goggles. Even with optimizers for XR device workloads, latency issues caused by high memory requirements still exist. To address this, Yang et al. [70] employ a 3D stacking neural network accelerator architecture to evaluate energy and latency improvements. The simulator provides estimates of various hardware metrics, such as execution time (latency), energy, input/output flow, and resource utilization of prototype ML accelerators for running AR/VR models. Based on the analytical model of the simulator, operators are extracted, and expected performance indicators are calculated based on the energy table of the accelerator architecture to guide architecture design decisions for different ML models. Subsequent

TABLE III

RESEARCH ON ENABLING TECHNOLOGIES IN THE INDUSTRIAL METAVERSE.

| Tech. | Ref. | Year | Scenes | Objectives | Methods | Limitations |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| $\mathrm{BC}$ | 18 | 2022 | Data collection to <br> storage. | Ensure credible execution <br> of Industrial Metaverse. | A BC-based trusted collaborative governance system <br> for Metaverse smart manufacturing scenarios. | Lack of simulation. |
|  | [20 | 2022 | Cross-enterprise <br> collaboration. | Improving privacy pro- <br> tection in the Industrial <br> Metaverse. | Combining joint learning and cross-chain techniques <br> to design a hierarchical BC architecture with main <br> chain and sub-chains. | Lack of mechanisms <br> to prevent malicious <br> local devices. |
|  | 19 | 2022 | Access manage- <br> ment. | Responding to Metaverse <br> and decentralized Internet <br> initiatives. | Dedicated frequency bands in regional spectral bands <br> for link transmission. | Lack of simulation. |
|  | 21 | 2022 | Cross-enterprise <br> collaboration. | Optimizing resource shar- <br> ing and utilization. | A practical BC-based MEC platform is proposed for <br> resource sharing and optimal utilization to accom- <br> plish requested offloading tasks. | No detailed assess- <br> ment of the proposed <br> system is provided. |
| DT | 36 | 2022 | Cross-enterprise <br> collaboration. | Joint design of commu- <br> nication, computing and <br> storage resources. | The latency minimization problem is formulated by <br> optimizing edge task offloading, arithmetic overhead, <br> and thus improving the QoE of DT in the Metaverse. | The bias is assumed <br> to be fixed. |
|  | 35 | 2022 | Base station data <br> receiving. | Ensuring the quality of <br> VSP's services. | The authors investigate the multi-base station chan- <br> nel resource allocation problem in a Metaverse DT <br> with an end-to-end cloud three-layer framework. | Storage resources are <br> not considered. |
|  | 29 | 2022 | Surface defect <br> detection. | DT solutions for small <br> surface defect inspection <br> tasks. | Integration of multimodal data. | Algorithms struggle <br> to handle massive <br> data scenarios. |
|  | 33 | 2022 | Base station data <br> receiving. | Reduce cloud load with <br> lightweight DT integration <br> with MEC. | A Service Placement Algorithm Based on Merkle <br> Tree. | The baseline <br> algorithm is not <br> enough. |
|  | 37 | 2022 | Crane <br> application. | Improved efficiency <br> of XR application <br> development.  | The architecture includes a perception layer and <br> ROS-based simulation model, and a Unity-based <br> service layer to create XR-based digital applications. | It may not be possible <br> to apply it to otheI <br> areas. |
|  | 40 | 2023 | Supply chain and <br> O\&M. | Promoting synergy <br> between DT and XR. | The TwinXR approach is proposed, which combines <br> information management DT and knowledge-based <br> XR. | Feature mapping is <br> entirely dependent on <br> shared data structure. |
| $6 \mathrm{G}$ | 4 | 2022 | Computer vision. | Organic integration of <br> edge AI and Metaverse <br> with 6G support. | Three novel edge Metaverse architectures are in- <br> troduced that use 6G-enabled edge AI to address <br> resource and computational limitations. | Missing network la- <br> tency and privacy is- <br> sues with $6 \mathrm{G}$. |
|  | 64 | 2022 | Product decision <br> and <br> service <br> phase. | Allows Metaverse mobile <br> users to download virtual <br> world scenes in real time. | An environment with multiple cellular stations is <br> designed where the task of downloading graphics <br> would be handed over between cellular stations. | The role of AI for $6 \mathrm{G}$ <br> is not addressed. |
| $\mathrm{XR}$ | 54 | 2018 | Hologram mod- <br> eling. | Developing agent-based <br> AR/MR systems. | Support for modifying agents in AR using different <br> heterogeneous agent programming techniques. | Lack of integration <br> with standards. |
|  | 65 | 2022 | Interaction \& As- <br> sistant. | Adapting to MTMM <br> workload requirements <br> for device heterogeneity. | Various features of real-time MMMT ML workloads <br> are discussed, and XRB ENCH and three represen- <br> tative executions are introduced. | No consideration of <br> cascading pipeline <br> complexity. |
|  | 66 | 2022 | None. | Understanding the rela- <br> tionship in XR's percep- <br> tion of user privacy. | The article analyzes the gap between user privacy <br> perceptions and their specific behaviors across XR. | Omission of audit and <br> transparency. |
|  | [56 | 2022 | Automated qual- <br> ity tests. | Operating system level <br> XR architecture. | Get better performance by enforcing fine-grained <br> control. | No mention of actual <br> use cases for XROS. |
|  | 67 | 2022 | Workload <br> metrics. | Correlation of AR to over- <br> all workload. | NASA-TLX for measuring subjective workloads. | Subjective <br> experiment. |
| $\mathrm{AI}$ | 68 | 2022 | Multi-Team col- <br> laboration. | Lightweight Metaverse <br> framework to support <br> language and semantic. | Summarizes common AI technologies and appli- <br> cation development templates, as well as common <br> functional modules and interfaces. | Lack of quantitative <br> assessment. |
|  | 69 | 2022 | Product design. | Improve 3D modeling ef- <br> ficiency. | AI-assisted VR modeling. | Lack of baseline com- <br> parison. |
|  | 70 | 2022 | Product design. | Memory interfaces are too <br> restrictive in terms of <br> bandwidth. | Higher XR resolution can be achieved by stacking <br> multiple layers of memory <br> using 3D stacking. | Lack of practical ex- <br> amples for deploy- <br> ment. |
|  | 71 | 2022 | Digit recognition. | Mitigating security risks <br> in the Metaverse. | Arming smart contracts with intelligence and using <br> AI algorithms for prediction. | Missing energy over- <br> head. |
|  | 72 | 2022 | Product <br> performance <br> analysis. | Improving the reliability <br> of numerical modeling. | Four ML methods are used to fabricate advanced <br> DTs of $42 \mathrm{SiCr}$ alloy considering all their uncertain- <br> ties and nonlinearities. | The industrial scenar- <br> ios involved are rela- <br> tively homogenous. |

work can further expand the memory capacity in the $\mathrm{X}, \mathrm{Y}$, and $\mathrm{Z}$ directions to identify points of diminishing returns and consider the practical deployment and implementation of the platform. In addition, for VR performance estimation, Yang et al. [74] contribute a dataset (VRQ-TJU) that can be used for VR quality assessment and introduced an end-to-end 3D CNN for video quality assessment. The 3D CNN takes VR differential video patches as input and considers information between different frames. Subsequent work can analyze various features rather than just spatial distribution changes. In addition to auxiliary architecture optimization and quality assessment, AI is widely used for VR image denoising, superresolution generation, etc., to provide operators with smaller granularity of operations and clearer interactive interfaces [75], [76].

Another trending direction entails AI-assisted biometrics for human pose recognition in complex environments and user device verification within the Metaverse's realm. The former encompasses motion recognition and body part tracking facilitated through the synergy of controllers and motion sensing devices. Such applications commonly manifest in player interaction scenarios within gaming contexts, but fall outside the purview of this discourse. The latter, primarily rooted in user privacy protection, will be expounded upon in the ensuing section.

In conjunction with the integration of XR, the Metaverse architecture empowered by BC, Web 3.0, and MEC has engendered the emergence of Decentralized AI (DeAI) [77]. In contrast to Centralized AI (CeAI), which embodies a static, centralized system leading to limited flexibility and scalability, DeAI possesses the capability to decompose tasks assigned by central departments into subtasks, subsequently allocating them to the next tier of agents for processing. Through collaborative efforts, these agents ultimately converge to formulate the final solution. While intricate AI models often entail substantial computational overhead, rendering real-time execution on edge devices within the Metaverse a challenge, lightweight AI processing solutions have matured. These encompass techniques such as parameter pruning, quantization, low-precision approximation, and knowledge distillation. Lee et al. [78] have successfully applied a lightweight LSTM model in Head-mounted Displays (HMDs). The network takes a set of 10 data points sampled from motion data at intervals of 5 timestamps as input. Subsequently, online learning techniques can be applied for personalization to enhance accuracy. By virtue of lightweight edge deployment, DeAI enables the realization of intelligent device agents at the edge level. Nodes within the edge layer undertake preliminary solutions through local reasoning and subsequently coordinate and communicate with neighboring nodes, thereby facilitating resource sharing. This paradigm presently represents the optimal approach for the Industrial Metaverse.

Another facet of AI-enabling technology revolves around the advancements within AI technology itself. With the escalating popularity of generative $\mathrm{AI}$ models such as $\mathrm{AI}$ generated content (AIGC), the capacity of AI to generate content has emerged as a new frontier within the Metaverse. Consider an employee equipped with an AR device tasked with arranging equipment in a virtual environment. In this scenario, the employee may require a catalog of pre-ordered equipment, prompting the input of relevant details into an AIGC model. The model then generates equipment entities based on the given context and scenario. Notably, AIGC within the Metaverse should meet two essential criteria: first, it should support device access at any time and any location; second, it should maximize user utility within the Metaverse while satisfying user needs. Nevertheless, due to variations in task suitability among AIGCs and the non-uniform distribution of computing power across edge servers, a resource scheduling scheme that ensures reasonable allocation becomes imperative. Addressing this concern, Du et al. [79] propose an optimal decision algorithm for AIGC based on a diffusion model, used to generate ASP selection decisions. The system collects AIGC task requirements from users, where user submissions include text prompts describing the desired content. Edge servers process and schedule these requests, ensuring efficient resource utilization and optimal system performance by understanding the specific requirements of each task, including computational complexity and output quality. After collecting user requirements, the system conducts semantic analysis through a knowledge graph to identify similarities and differences between user prompts. This approach provides an effective solution for deploying AIGC services in the Metaverse. Similarly focusing on edge deployment for AIGC, $\mathrm{Xu}$ et al. [80] synthesize the trade-offs between reasoning latency, resource consumption, and accuracy by managing pretrained foundation models (PFMs) on edge servers using the minimal context algorithm, reducing the overall system cost. This work provides a formal mathematical expression for basic model caching and inference problems and the minimal context algorithm. Overall, the emergence of AIGC illuminates new avenues of development within the Metaverse, enhancing content diversity and alleviating repetitive production labor. However, it also gives rise to novel challenges, such as the regulation of generative content and the screening of potentially explosive content [81].

AI is poised to enable intelligent networking, immersive digital worlds, inclusive user interfaces, accurate avatar creation, multilingual accessibility, and many other features within the concept of the Metaverse, which is inherently usercentric [82]. The more precise and refined the AI models are, the better the overall experience can be for users. Multilingual accessibility and automatic translation are also critical components that facilitate cross-border enterprise access to the Metaverse [83]. Nevertheless, interpretable AI is essential to enhancing brain mapping accuracy, as it enables explanations for trained models and enhances the interpretability of AI black box operations. In recent years, a variety of interpretable AI-based approaches have been introduced to improve control and monitoring of unwanted or adverse effects, such as biased decision making. Interpretable AI utilizes semantic representations to uncover relationships between different object coverages in images, thereby enhancing the interpretability of deep learning solutions. Additionally, interpretable AI can be utilized for product design and marketing, helping to analyze competitor performance and identify opportunities

![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-15.jpg?height=369&width=1228&top_left_y=184&top_left_x=446)

Fig. 8. Existing issues in Industrial Metaverse. Including security and privacy, resource allocation, and interoperability. In terms of security, it can be discussed from the perspectives of sensors, platforms, and enabling technologies themselves. Resource allocation includes two aspects: computing resource allocation and storage resource allocation.

to outperform them [84]. By leveraging customer-generated reviews, interpretable $\mathrm{AI}$ can extract competitive factors for products and effectively reflect customer opinions.

2) Summary: AI technologies in the Industrial Metaverse generally require meeting requirements such as real-time capability [85], scalability [86], and interpretability [87]. However, there is still a significant research gap in interpretability. Many AI-assisted services and applications in the Metaverse rely on AI agents that are driven by black-box ML models, lacking explanation and interpretability capabilities. Developers, virtual world designers, and users of the Industrial Metaverse cannot understand the AI decision-making process from a mechanistic perspective. The lack of interpretability may lead to potential risks, as even a $1 \%$ error rate in decision-making can be unacceptable in industrial manufacturing. This impact not only affects economic benefits but also safety concerns.

## III. EXISTING ISSUES IN INDUSTRIAL METAVERSE

In this section, we will delve into the crucial issues that currently exist in the Industrial Metaverse. With the rising complexity of connections and interactions between systems and devices, security threats and vulnerabilities have become a more pressing concern. Furthermore, the vast number of sensors, devices, and data involved in the Industrial Metaverse require efficient data management to prevent privacy breaches during data transmission [88]. Given that most enterprises prioritize profit maximization and face resource constraints, scientific resource optimization is necessary to reduce costs. We will elaborate on these challenges in the following sections. Fig. 8 illustrates the existing issues in Industrial Metaverse.

## A. Security and Confidentiality

In the preceding segment, we presented an overview of how the Industrial Metaverse has attained extraordinary immersive encounters through the facilitation of cutting-edge technologies such as XR, DT, and AI, thereby establishing a novel paradigm for industrial administration. However, at the same time, it has also created unprecedented security threats. From data collection to the final decision-making stage, security and privacy challenges pervade the entire production cycle of the Industrial Metaverse. For example, the requirements for data integrity and availability, which are based on industrial data itself (i.e., ensuring that data flows from sensor collection endpoints to DT models without being maliciously modified or destroyed), as well as the requirements for authentication/authorization and traceability for employees (i.e., strict verification mechanisms for device operations and data access to ensure that only legitimate users can access them).

Although in the previous section, we mentioned the use of BC technology for anonymizing data and user identities through techniques such as identity verification and smart contracts, this alone cannot solve all security issues. In the following, we will organize existing solutions to security and privacy issues existing in various stages of the Industrial Metaverse based on the architecture diagram provided in Fig. 2 This includes data sources collected at the sensor end, as well as data and network security threats during subsequent cross-component transmission processes. We will also discuss security issues within components such as AI, BC, and XR, and further discuss possible research directions.

1) Sensor-side Industrial Data Source: To enhance the immersion of the Metaverse and achieve more precise industrial decision models, industrial sensors play a pivotal role in collecting extensive data for the platform. These data sources encompass not only initial training samples and physical parameters used to construct the virtual space but also user information.

When it comes to industrial source data, the emphasis lies in safeguarding the data itself. For instance, redundant information in visual content, which does not contribute to the training of models but is indiscriminately collected by sensors, can be obfuscated by overlaying a mask to mitigate potential privacy breaches. Validated operators can have their virtual images in the Metaverse replaced with generic mappings. Another approach involves employing black and white list mechanisms, where only specific information is retained, and any data outside the whitelist is immediately discarded and inaccessible. Wu et al. [89] propose a data misuse prevention mechanism called DAPter, wherein input data undergoes lightweight generator transformations to remove unnecessary information with minimal additional overhead. The blacklisting mechanism is commonly utilized for safeguarding user facial information. In this method, faces are decomposed into vector signals, comprising identity and attribute features. Subsequently, an anonymization model is generated to integrate and reconstruct the vectors based on the need for further processing [90]. This approach can also be employed for classifying features of in-
dustrial devices, such as device models and metric parameters. Additionally, traditional data encryption and storage schemes used in cloud-side end systems can be applied to Industrial Metaverse platforms. These include classical symmetric and asymmetric encryption techniques, cloud storage encryption, disk encryption, and more.

Apart from the issue of data leakage, the quality of data sources is also classified as a data security concern. On one hand, malicious users may generate low-quality or even false data to disrupt virtual world project construction. On the other hand, profit-oriented participants may lower data verification standards to control costs, leading to misleading data analysis and model training. The first case can be addressed through identification of malicious users and tackling datarelated attacks. Common data attacks include data injection attacks and data tampering attacks. Data injection attacks involve perpetrators injecting false data into industrial systems through illicit means to mislead and disrupt system operations. Countermeasures against such attacks include implementing detection and verification mechanisms, as well as access control. For example, Pang et al. [91] employ a Kalman anomaly detector based on residuals to securely control networked systems, while Guo et al. [92] use a random coding scheme to detect residuals, transforming the design problem of coded signals into a constrained nonconvex optimization problem. Access control, which restricts data modification to authorized personnel only, is widely employed in industrial settings. Existing schemes include access control architectures based on BC [93], cryptographic storage methods [94], and more. These access control schemes can also be applied to combat data tampering attacks. As for the second case, various trustworthiness identification and processing schemes have been established to address the issue of low-quality data sources in industrial settings. These schemes encompass sensor node trustworthiness tracing [95], mining association rules using data mining algorithms [96], methods for prognosis of lowquality data integration [97], and data enhancement techniques 98].

2) Data and Network Security Threats in Distributed Industrial Platforms: The distributed architecture of the Industrial Metaverse platform is well-suited for handling large-scale data processing scenarios. However, the introduction of massive P2P links also brings about increased risks of data leakage and management challenges. On one hand, the lifecycle of the Industrial Metaverse spans across different stages, with multiple dispersed stakeholders performing various operations. The increase in digital connections may expose enterprise assets or production processes to data leakage risks. However, leveraging BC-based distributed storage can effectively manage data throughout the PLC. The storage layer provides secure distributed data storage with lightweight and scalability features. Suhail et al. [99] discuss deployment schemes for BC-based Metaverse platforms. They utilized multiple sensors with overlapping perspectives for cross-validation to ensure the credibility of on-chain data. Empowering decisionmaking through BC in various industrial use cases addresses challenges such as different data repositories and insecure propagation. Peng et al. [100] further consider edge offloading in industrial cloud-edge collaborative architectures, utilizing DT to capture temporal features of computational resources. Additionally, in the second incentive phase, they integrated privacy-security investments to assist industrial agents in making offloading decisions. Shen et al. [101] integrate distributed storage and cloud storage, encrypting raw virtual data in the cloud, storing transaction records in the $\mathrm{BC}$, and maximizing overall benefits for manufacturers and buyers through optimal sampling selection. However, for industrial sensors with higher sampling rates, limitations of real-time sharing need to be further considered. Furthermore, BC is not a one-sizefits-all solution, as issues such as decision paralysis due to information overload, high energy consumption, and latency may arise. Moreover, the nature of the "black box" from BC compels workers to trust the integrity and fairness of processes without understanding the technical basis.

Another issue arising from the distributed architecture of the Industrial Metaverse is the potential for errors and attacks due to multiple untrusted but interdependent participating entities and the integration of non-intersecting data from multiple sources [102]. For instance, man-in-the-middle attacks or altering communication between users and servers may intercept/modify data packets, compromising the confidentiality, integrity, and availability of the system. Cross-site scripting attacks inject external malicious JavaScript code into websites, enabling attackers to inject malicious JavaScript code directly into client websites or have client websites request storage locations. Immersive interaction involves many devices such as wearable devices, headsets, base stations, and controllers, with large-scale data exchange. Despite manufacturers implementing encrypted communication, industrial data may still be accessed by adversaries through eavesdropping methods during transmission. Currently, efforts are underway to enhance industrial network security and anomaly detection. Kim et al. [103] utilize a system based on autoencoders and long shortterm memory models to learn normal values of sensors and actuators for monitoring network attacks. Kwon et al. [104] study distributed network protocols widely used in distribution and transmission systems, proposing an anomaly detection method based on bidirectional recurrent networks and IEEE 1815.1. They verified the method with attack data from networked physical systems, successfully detecting eight types of attacks, including unauthorized command attacks and false data injection. However, existing detection systems deployed in Metaverse platforms often result in high computational costs and delays and are difficult to adapt to dynamic attack environments (such as atypical and polymorphic attacks). Therefore, network security researchers must further optimize and promote existing strategies to establish effective models.

3) Inherent Security Vulnerabilities of Enabling Technologies: The enabling technologies mentioned in Section II of the Industrial Metaverse also encompass significant security vulnerabilities that cannot be disregarded.

Concerning AI technology, although Wang et al. [105] highlight its potential for detecting illicit entities and anomalous accounts within the Metaverse, such as discerning suspicious signals through correlating user-generated textual information and spatiotemporal activities, AI also poses various potential
risks $[106]$. On one hand, the industrial platform interconnects a wide array of industrial devices, Metaverse users, collaborative enterprises, and applications, necessitating automated regulation and decision-making due to the intricacies of data management. This grants algorithms extensive operational privileges to reduce managerial costs. However, this approach may give rise to issues like biased decisions [107], low transparency [108], and operational fragility [109]. While certain existing work suggests hierarchical governance structures for algorithm accountability or assesses potential risks of algorithm engines based on AI explainability, a unified compliance governance protocol is yet to be established. On the other hand, due to the black-box nature of AI, malevolent attackers can infiltrate the enterprise source code with malicious code to cause system-level disruption. For instance, they may employ Generative Adversarial Networks (GANs) to fabricate false virtual data or embed malicious software during the model training process, which developers often find challenging to detect in early training results. Therefore, in the industrial context, modular interpretable AI may serve as a promising avenue for future research.

$\mathrm{BC}$ records user information and production data in the Industrial Metaverse platform through a distributed ledger and facilitates anonymous P2P interactions through encryption, providing dependable data storage and transmission for enterprises and users [110]. Nevertheless, BC also harbors inherent security risks. Some emanate from the consensus mechanism, such as PoW, wherein an attacker with over half of the BC's computing power can directly seize control of the entire platform system. Additionally, while consensusbased BC endows robust tamper resistance for industrial data, the computational overhead generated by existing mainstream consensus mechanisms is also a noteworthy concern. Another set of risks arises from BC's key distribution mechanism. For instance, its decentralized key management implies that private keys might be concentrated and stored on specific entities or servers, rendering all keys vulnerable in the event of an attack on the entity. To avert private key leakage, multi-signature mechanisms can be employed, along with offline storage to mitigate network attacks.

Furthermore, XR-based Industrial Metaverse interaction technologies are susceptible to hacker attacks due to their numerous data interfaces. For example, various systems in factories utilize identity authentication methods such as fingerprint recognition and facial recognition for authorization. Hackers can gain access to device privileges by learning and reconstructing facial information. One immediate countermeasure is the design of novel authentication mechanisms. For instance, Wang et al. [111] devise an identity authentication scheme based on palmprints, which verifies through collecting the reverse scattered signal of acoustic signals, circumventing privacy leakage risks associated with image recognition. Duezguen et al. [112] leverage the concept of semantic connections within the ZeTA protocol to support identity authentication in XR context sharing. Finally, the authors evaluated the correlation between different interaction modes and risks. In the future, adopting compound multi-dimensional authentication technologies, coupled with timely firmware updates, may effectively mitigate privacy breaches during the interaction process. Lastly, beyond the aforementioned issues, addressing security challenges in the Industrial Metaverse necessitates keeping abreast of relevant legal documents and enterprise standards.

## B. Resource Allocation

Due to the increasing number of applications and industrial devices involved in industrial production, distributed deployment of the Metaverse platform may lead to performance imbalances among subsystems and processors, such as computing power and throughput. When facing various types of tasks throughout the PLC, enterprise demands are often dynamic and diverse, posing challenges to the management of network and computing resources. Below, we discuss resource allocation strategies in the Industrial Metaverse focusing on computing resources and storage resources.

1) Computing Resource Allocation Issues: The Industrial Metaverse demands significant computational resources to seamlessly render 3D virtual environments. However, due to the limited computational capabilities of certain lightweight devices within factories, they may fall short for computationally intensive rendering tasks. Although cloud servers offer sufficient computing power for remote rendering, the current state of cloud infrastructure may struggle to meet the ultra-low latency demands of industrial production activities [113].

In collaborative multi-enterprise Industrial Metaverse platforms, some adaptable traditional distributed technologies are gradually being introduced. These include MEC and FL [114]. FL, particularly, is recognized for its privacy-preserving characteristics and excellent scalability as a distributed architecture that aggregates models remotely without exposing data. AlQuraan et al. [115] discuss the potential applications of FL in various wireless networks, such as visible light communication and massive multiple-input multiple-output (mMIMO), to maintain network performance stability by training models that alleviate microbase station congestion. Nevertheless, despite offloading computation to the edge, FL may still face challenges in frequent information exchange between cloud servers and edge nodes due to the continuous generation of large volumes of new data in industrial production processes. Additionally, the increasing number of parameters and limited bandwidth resources may become bottlenecks for FL. Furthermore, due to differences in device models and performance in factories and varying requirements for model accuracy and computational resources in different scenarios, it is difficult to consider a universal FL aggregation design solution. Furthermore, the number of nodes in Industrial Metaverse systems is often indeterminable during application deployment, necessitating dynamic adjustments in resource allocation to accommodate the addition or removal of nodes [116]. In the face of extreme scenarios, the robustness of the deployment architecture must also be taken into consideration.

Another common architecture is based on the allocation of computing resources in MEC. MEC, by deploying computing nodes near the terminals, is particularly suitable for highconcurrency industrial scenarios. Since edge computing nodes
are close to physical devices, they can continue local data processing and decision-making even in cases of unstable or disconnected network connections [117]. Tan et al. [118] introduce DT into edge networks and designed power optimization based on Deep Q-Networks (DQN) for dynamic channel conditions in MEC. The proposed architecture was validated through simulations for applications in automated warehousing logistics and mobile production lines. In Industrial Metaverse platforms, VSPs are responsible for providing resources ordered by various manufacturers. However, excessive resource allocation may lead to resource over-provisioning issues. $\mathrm{Ng}$ et al. [119] propose a decision framework for VSPs in the Metaverse based on two-stage stochastic integer programming to minimize overall network costs considering the uncertainty of user demands. However, they did not consider solutions for resource demands in the case of large-scale resource requirements. To address these challenges, Chu et al. [120] decompose the Metaverse platform and utilized application similarity to improve resource utilization. They developed a self-learning framework based on Markov processes to capture implementation characteristics of the Metaverse. Considering the dynamic industrial environment, Yu et al. [121] propose an end-to-end self-organizing resource allocation framework under a 5G heterogeneous network, greatly enhancing Quality of Service (QoS).

In scenarios involving multiple for-profit enterprises, the utility of VSPs in the Industrial Metaverse is considered a function of latency and resource payment costs induced by resource offloading. When the utility is non-negative, VSPs will be willing to invest more computational resources in enterprise collaboration [122]. One scheduling solution involves designing incentive mechanisms to maximize device utilization. Li et al. [123] introduce a distributed crowdsourcing framework called CrowdBC, where smart contracts are used for automated execution, and $\mathrm{BC}$ is used for user reputation management and incentive mechanisms. Reputation reflects the past performance of enterprise employees in task resolution. After submitting the execution plan, enterprise employees can request the system to perform task evaluation and record it. Luong et al. [122] propose using multi-agent auctions as an incentive mechanism to support semantic communication (SemCom) and computing resource allocation in the Metaverse. Transmitting semantic information instead of raw data can further reduce offloading latency. Additionally, as the key to synchronizing the real world with the virtual world lies in the data collected by IoT devices and sensors, Han et al. [124] design a dynamic resource allocation framework using a hybrid evolutionary dynamics approach to synchronize the Metaverse with IoT services and data. However, the aforementioned works did not consider the dynamic characteristics of each computing node in the Industrial Metaverse system, such as smart handheld terminals and AGVs. When nodes are predetermined, idle computational resources may remain underutilized when MEC handles computationally intensive tasks.

2) Storage Resource Allocation Issues: Due to the fundamental characteristics of the Metaverse, such as diverse applications, heterogeneous connected devices, ubiquitous user access, and digital identities, in addition to computational capacity, a sustainable storage platform and efficient storage strategy are urgently needed. Servers will collaborate to store information and share content with other servers. In this sense, the payload of redundant data processing and uploading can be reduced by dynamically obtaining terminal information, predicting its demand, and collaboratively storing data, thereby improving the utilization of storage resources and reducing the service latency of the entire system. However, while storing sensor data in the cloud may be a viable solution, as mentioned earlier, cloud-based distributed storage requires continuous internet connectivity and becomes unavailable when the platform encounters emergencies due to internet connection disruptions. Therefore, Rashid et al. [125] proposed the idea of using private edge devices for decentralized distributed data storage in emergencies. They developed a game theory model to dynamically incentivize resource allocation and regulate the QoS of the network. Devices contributing more storage space will receive better QoS (i.e., network bandwidth). To address edge device configuration issues, the authors developed an application that seamlessly runs in Docker containers and connects to the EdgeStore API for system configuration. EdgeStore enables maximizing system availability in emergencies by establishing cooperative storage sharing plans and storing data on private edge devices.

Storage functionality typically includes content storage and policy storage. In the policy storage scheme proposed by Zhao et al. [126], when the server receives relevant requests, it processes tasks based on the request and then returns them to the selected policy device terminal. In Liu et al.'s [127] storage scheme, content is pre-stored on servers. When relevant requests arrive, the server can directly return the data to the terminal. Overall, content storage has lower latency. However, the policy storage scheme can better generalize and adapt to heterogeneous applications. It can support different data requests from users well through preset policies.

Unlike the stable resource scheduling in cloud centers, edge servers are less inclined to share storage space and data due to specific requirements such as workload, latency, and privacy [128]. In emergencies, the inconsistent availability information of storage resources on edge servers may lead to a high data loss rate. To address this issue, Vaquero et al. [129] propose a method to replicate data on all nodes. However, the above method will increase the overall cost of the system and may degrade performance. Recent research work also involves edge caching technology in Metaverse scenarios. Cai et al. [130] design an efficient control policy that integrates computing, caching, and communication (3C) resources and implements the policy by implementing a multi-pipeline traffic control and 3C resource scheduling mechanism. To further optimize the online delivery performance of data-intensive services, the authors also proposed a database placement strategy based on throughput maximization and two effective database replacement strategies. On the other hand, Huynh et al. [131] propose an innovative DT solution that combines MEC and URLLC to support Metaverse applications. From the perspective of the $3 \mathrm{C}$ integration model, this solution optimizes edge caching policies, task offloading strategies, and
the allocation of computing and communication resources to address the latency optimization problem in the Metaverse enabled by DT.

## C. Interoperability

As a complex system, the Industrial Metaverse platform integrates various sensors and control subsystems with different hardware architectures and control components. If there is no support for interoperability, user identities and industrial equipment data will be limited to specific VSPs, contradicting the original intention of an interconnected Metaverse [132]. Therefore, interoperability of subsystems within the platform and interoperability across platforms are important considerations. Currently, solving the interoperability issues of the Industrial Metaverse mainly starts from two aspects: identity and identification authentication, and interface and integration framework.

Identity and identification define unique codes for users and production equipment, while binding the identification with attributes such as device status and human behavior [133]. This forms the foundation for achieving interoperability. These attributes can be provided by physical devices in factories or created by enterprise personnel themselves. In different platforms, the same identity and identification correspond to the same static attributes. Gadekallu et al. [134] suggest that $\mathrm{BC}$ can facilitate the exchange of data between different subMetaverses. They proposed using cross-chain protocols to exchange data on multiple BCs located in different virtual worlds. Therefore, Li et al. [133] propose a MetaOpera, a BCbased interoperability protocol for the Metaverse. It supports cross-chain technology and on-chain/off-chain techniques for the interaction between MetaOpera and decentralized platforms as well as top centralized platforms. Compared to authentication technologies based on Sidechains [135], this method achieves lower interaction latency. Additionally, Ghirmai et al. [136] propose using Self-Sovereign Identity (SSI) technology, which replaces traditional identity verification with distributed identity verification using zero-knowledge proofs. For example, two workers docked on a Metaverse platform can prove each other's public keys according to the principles of SSI and encrypt their interaction records using pre-shared keys. Each worker holds a unique NFT as a unique identifier. However, the authors also pointed out that the issue of BC storage should not be ignored because controlling and managing data and identities may require different storage mechanisms. Additionally, their scenario only applies to static attributes in the Metaverse and does not involve behavioral objects.

Split learning, as a distributed architecture, involves multiple devices and servers in an industrial platform [137]. It can split the backbone network into multiple sub-networks, execute multiple sub-tasks in parallel on different devices or servers, and then integrate the results. Inspired by this, Rawal et al. [138] introduce a Split-Protocol that provides interoperability services. They assume that critical elements and functions are provided by a meta-server, and the system consists of ten identical meta-servers, each executing specific functions of the Metaverse. Role transitions can be achieved through the Split-Protocol. This protocol with role-switching capability can improve the availability of the Metaverse system. Furthermore, Hashash et al. [139] emphasize the coordination between DTs and Edge Metaverse. A Physical Twin (PT) system running in a large-scale sensing area is replicated as a Cyber Twin (CT) system on MEC servers. The area is divided into distributed subgrids and transmitted remotely to MEC servers. Their designed interactive algorithm utilizes optimal transport theory, considering both computational power and data synchronization issues.

In summary, each Industrial Metaverse platform has different configurations, and there is still a lack of standard development patterns. Interoperability and compatibility are guarantees for providing a better operating experience in the Industrial Metaverse. Cooperation between companies and industries is needed to increase the content and value of the Industrial Metaverse and establish an open collaborative ecosystem [140].

## IV. StandARdiZATion

Since the concept of the Metaverse was proposed, efforts have been made to promote its industrial deployment and ensure interoperability and compatibility between different Industrial Metaverse platforms, as well as to reduce the cost of industrial system development and maintenance. Standardization work in this regard has been ongoing.

The MPEG-V standardization, initiated in 2008, defined the exchange of data at the interface between the real world and the virtual world, with the aim of standardizing 4D sensory effects and the virtual world [143]. The standard consists of seven parts, among which 23005.1 provides an overview of the architecture and specifies the relevant information representation for achieving interoperability. 23005.2-3 covers syntax and semantic information, including sensory device functionality and sensor functionality. 23005.5-6 standardize data formats and data types, which are particularly important in industrial multimodal data scenarios [144]. Subsequently, the IEEE Standards Association launched the P2888 project (Interface Networking and Physical World) and established the Synchronization Metaverse as a supplement to the ISO/IEC 23005 series of standards. IEEE 2888 includes four parts: specifications for sensor interfaces in networking and physical worlds, standards for actuator interfaces in networking and physical worlds, orchestration of digital synchronization between networking and physical worlds, and architecture for VR disaster response systems. The first three parts aim to provide general technology for DT or the Metaverse, while the fourth standard aims to provide specific applications for the above three standards [145].

In June 2022, the Metaverse Standards Forum (MSF) was initiated and established by the Khronos Group, with participants including Meta, Microsoft, Epic, NVIDIA, Qualcomm, Sony, and other 37 technology giants engaged in the field of the Metaverse, covering industries such as chip manufacturers and game companies. The organization has now gathered over 2,400 industry vendors and standard organizations to

TABLE IV

RESEARCH ON EXISTING ISSUES IN THE INDUSTRIAL METAVERSE.

| Challenges | Issues | Ref. | Algorithm / Framework | Contribution / Performance | Year |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Security <br> $\&$ <br> Confidentiality | Sensor-side <br> Industrial <br> Data Source | $\|89\|$ | A data misuse prevention mech- <br> anism called DAPter. | DAPter can substantially raise the bar of the data abuse <br> difficulty with little impact on QoS and overhead. | 2021 |
|  |  | $\mid 91$ | Kalman anomaly detector based <br> on residuals. | This may be helpful for guaranteeing the secure control <br> of a networked system by protecting partial critical sensor <br> measurements from FDI attacks. | 2022 |
|  |  | $\mid 92]$ | A random coding scheme to de- <br> tect residuals. | Estimation-optimization iteration algorithm is to obtain a <br> numerical solution of the coding signal covariance. | 2022 |
|  | Data \& Network <br> Security | \|100 | A two-stage incentive mecha- <br> nism. | Efficient resource allocation and computation offloading <br> are realized while privacy is guaranteed. | 2023 |
|  |  | $\overline{101}$ | BC and DT-based distributed <br> storage and cloud storage. | The algorithm is better than traditional method for maxi- <br> mizing the total social benefits. | 2021 |
|  |  | $\mid 103$ | Two payload-based anomaly de- <br> tection methods. | The accuracy and the FP rate are superior to the remaining <br> methods. | 2020 |
|  |  | $\mid \overline{104}$ | An intrusion detection system for <br> an IEEE 1815.1-based power sys- <br> tem network. | Five types of CMB attacks and three types of FDI and <br> DR attacks are successfully detected. | 2020 |
|  | Inherent Security <br> Vulnerabilities of <br> Enabling Technologies | $\overline{111}$ | AcoPalm, a palm print recogni- <br> tion system. | AcoPalm is resistant to replay and mimicry attacks, and <br> can achieve $96.22 \%$ authentication accuracy. | 2022 |
|  |  | $\lceil\overline{112}]$ | Zeta protocol. | The authors explain how it can be used with the available <br> interaction methods offered by head-mounted displays. | 2020 |
| Resource <br> Allocation | Computing Resource <br> Allocation Issues | $\overline{\mid 118}]$ | A framework for MEC based on <br> networked digital dual bases. | Web-based DT frameworks with AI capabilities can fur- <br> ther reduce task processing latency and improve QoS. | 2023 |
|  |  | $\mid \overline{119}$ | A stochastic optimal resource <br> allocation scheme based on <br> stochastic integer programming. | The proposed scheme minimizes the cost of VSPs while <br> taking into account the uncertainty of user demand. | 2022 |
|  |  | $\|\overline{120}\|$ | A framework based on semi- <br> Markov decision processes. | The proposed approach can realize up to $120 \%$ revenue for <br> Metaverse VSPs and a $178.9 \%$ probability of acceptance <br> for Metaverse application requests. | 2023 |
|  |  | $\overline{121}$ | Resource allocation mechanism <br> for IIoT in $5 \mathrm{G}$ heterogeneous net- <br> works. | It is possible to achieve better performance than other tra- <br> ditional deep learning (DL) methods and maintain quality <br> of service above accepted levels. | 2022 |
|  |  | $\mid \overline{122}$ | The design of the DL-based auc- <br> tion. | Auction improves the revenue while satisfying the indi- <br> vidual rationality and incentive compatibility constraints. | 2022 |
|  |  | $\overline{123}$ | A BC-based decentralized crowd- <br> sourcing framework. | A software prototype is implemented on the Ethernet <br> public test network with real datasets. | 2019 |
|  |  | $\mid 124$ | A hybrid evolutionary dynamics <br> approach. | A dynamic resource allocation framework to synchronize <br> the Metaverse with IoT services and data. | 2021 |
|  | Storage Resource <br> Allocation Issues | $\overline{125}$ | Edge-based distributed storage <br> system. | EdgeStore outperforms state-of-the-art distributed storage <br> systems in terms of throughput, energy consumption, and <br> latency. | 2019 |
|  |  | $\|126\|$ | ![](https://cdn.mathpix.com/cropped/2024_06_04_1e1ce86bbbb6315d30afg-20.jpg?height=136&width=394&top_left_y=1710&top_left_x=761) | The algorithm maximizes the number of requests served <br> by the low latency edge cloud server. | 2019 |
|  |  | $\mid \overline{127}$ | Cache-enhanced multi-user MEC <br> system. | The proposed joint optimization of caching, computation, <br> and communication methods can improve energy effi- <br> ciency at a lower time cost. | 2019 |
|  |  | 129 | Introduced next-generation or- <br> chestration technology. | Orchestration requirements for next-generation workflow <br> management in microservices architectures and IoT envi- <br> ronments are identified. | 2018 |
|  |  | $\mid \overline{130}$ | Designed the first throughput- <br> optimal control strategy. | The proposed novel multi-pipeline flow control and $3 \mathrm{C}$ <br> resource orchestration mechanism has excellent perfor- <br> mance. | 2022 |
| Interoperability |  | $\mid 141$ | A protocol called MetaOpera <br> generalized cross-Metaverse in- <br> teroperability. | The size of cross-global proofs and the average time <br> of cross-global transactions are reduced by $8 \mathrm{x}$ and $3 \mathrm{x}$, <br> respectively, using the proposed solution compared to the <br> Side-chain solution. | 2023 |
|  |  | $\mid \overline{136}$ | Self-Sovereign Identity (SSI) in- <br> tegrated with BC. | Helps address issues of decentralization, trust and inter- <br> operability in the Metaverse. | 2022 |
|  |  | $\overline{142}]$ | Introduceing an interoperable <br> splitting protocol. | The resilience and high-availability of a Metaverse system <br> for role switching. | 2022 |
|  |  | $\mid \overline{139}$ | Iterative algorithms for joint sub- <br> elements and DT associations on <br> MEC servers. | The proposed solution coordinates the interactions be- <br> tween the DT and the subuniverse, thus reducing the sub- <br> synchronization time by $25.75 \%$. | 2022 |

collaborate across the industry on building an open Metaverse, interoperability testing, open-source tools, deployment guidance, and more. In July of the same year, ISO/IEC JTC 1/SC 29 MPEG WG2 Market Need Ad Hoc Group (AHG) launched Metaverse case collection and technical analysis to identify MPEG efficient coding and related system standard technical requirements that can support future Metaverse experiences. In January 2023, the IEC Standardization Evaluation Group 15 on Metaverse (SEG 15) was established. Its specific work includes studying the standardization roadmap for the Metaverse field in conjunction with current standardization achievements and development trends, establishing timely contact with TC/SC/Syc (including JTC1) as well as ISO and other relevant organizations, and providing work proposals for the International Electrotechnical Commission Standards Management Board (IECSMB). Recently, the IEEE P3812 series of standards is in the drafting phase, which defines the requirements for the Metaverse identity framework [146]. The provided identity framework is intended for use across different Metaverse systems to meet the requirements for interoperability/interaction of entities across multiple Metaverse systems. Additionally, the IEEE P2048.101 series of standards define the general technical framework, components, integration, and major business processes for AR systems applied to mobile devices [147].

In the future, as Industrial Metaverse gradually takes off, the accompanying standards regarding interoperability, security and privacy, openness and scalability, cross-industry collaboration, and other aspects will become increasingly refined, forming a comprehensive standard system.

## V. FUTURE TRENDS AND OUTLOOK FOR INDUSTRIAL METAVERSE

While researchers have made significant efforts in studying the Industrial Metaverse tailored to industrial scenarios, alongside notable achievements, there are additional issues warranting consideration. In this section, we further expand our perspective and propose several future research directions for the benefit of researchers. Additionally, for each direction, we provide a brief list of skills/mathematical tools researchers should master, serving as foundational requirements for conducting related research:

## A. Flexible Communication Network Architecture

The seamless internet connectivity of the Metaverse largely relies on reliable broadband services, with holographic communication requiring real-time operation, sufficient bandwidth, and throughput [148]. While low latency and high reliability are the primary advantages of $5 \mathrm{G} / 6 \mathrm{G}$ services, consistent latency requires a flexible end-to-end architecture. This is because the Industrial Metaverse requires real-time communication of video, audio, and data. Currently, there is limited research on the network architecture and signal processing aspects of $6 \mathrm{G}$ in the context of the Metaverse. Relevant skills/mathematical tools: Wireless communication theory, signal processing, 5G/6G network architecture design.

## B. Privacy and Security Issues

In Section III, we have extensively discussed the confidentiality and security issues of the Industrial Metaverse from the perspectives of data, network, distributed platforms, and enabling technologies, as well as existing solutions. It can be foreseen that, as the Metaverse platform gradually matures, its security measures will largely rely on continuous upgrades through security patches. However, with the increasing prevalence of sophisticated cyber attacks, such security measures may bring challenges in terms of operational costs and scalability [149]. In the future, it is possible to adopt intrinsic security solutions, such as quantum key distribution, which utilizes channel-based keys generated through quantum entanglement to address information leakage in wireless transmissions [102]. Relevant skills/mathematical tools: Quantum key distribution, cryptography, network security.

## C. Lightweight and Ease of Use

Since Industrial Metaverse scenarios are constantly producing, storing, and transmitting data, they require massive access to hardware devices. These devices include production equipment deployed in factories, industrial sensor devices, and access devices such as VR glasses for user interaction. The lightweighting at the hardware level has been slow to materialize. Ways to reduce hardware overhead as well as increase the flexibility and scalability of devices is a valuable topic [150]. Relevant skills/mathematical tools: Embedded system design, hardware optimization, sensor technology.

## D. Model Accuracy and Completeness

The key factor affecting model accuracy in assembly line mode manufacturing applications is the quality of the data collected. The user must provide the system with complete and accurate available data [151]. Although the addition of DT technology can realize virtual samples directly into the model, the data sources often have serious heterogeneity due to different equipment models and data types, and the data fusion and calibration methods at this stage still need to be improved. Relevant skills/mathematical tools: Industrial data preprocessing methods, data fusion methods, machine learning algorithms.

## E. Unified Architecture

Central to the realization of the Metaverse lies the crucial foundation of Web 3.0 architecture. However, the intricate task of amalgamating diverse technologies within a unified architecture necessitates comprehensive efforts. In order to address this challenge, the development of common protocols capable of accommodating the distinct needs and requirements of various industrial sectors becomes imperative. Such protocols would serve as the bedrock for seamless integration, enabling interoperability and synergy among different technologies. By fostering a standardized framework, the development of the Metaverse can progress cohesively, accommodating a multitude of industries and catering to the unique demands of users across diverse domains [152]. Relevant skills/mathematical
tools: Network protocol design, distributed system architecture, Web technologies.

## F. Components are Fully Decentralized

The core feature of Web3.0 is decentralization, but APIs at this stage do not integrate well with Web3.0, and new decentralized common protocols need to be developed to achieve complete decentralization of Metaverse components [153]. Only the successful implementation of a decentralized network will occur. This is a challenge because web components and servers still run through legacy controls and protocols, and therefore the backbone infrastructure will need to be extensively changed [154]. Therefore, in the scope of the future, the design of decentralized protocols running on the web needs to be investigated, and open standards and APIs need to be integrated with Web 3.0 engines where data can be seamlessly transferred to different endpoints. Relevant skills/mathematical tools: Decentralized technologies [155], BC development, smart contract programming.

## G. Efficient Cross-chain Authentication

Efficient cross-chain authentication and governance is crucial to ensure the security and legitimacy of digital assetrelated activities (e.g., asset transactions) across different submetadomains built on heterogeneous BCs. Current crosschain mechanisms are mainly focused on digital asset transfers and rely on notary schemes, hash locks, relay chains and side chains, with few mechanisms considering cross-chain authentication and governance for meta-domains. The implementation, efficiency and security of cross-domain and BC authentication need further research. Moreover, novel decentralized, hierarchical and penetrating cross-chain governance mechanisms also need further research in the Metaverse. In addition, efficient meta-space-specific consensus mechanisms, redesigned block structures, and well-designed user incentives are all necessary for unique meta-space applications [156]. In summary, open challenges include application-specific governance rule design, programmable and scalable cross-chain governance architecture design, on-chain entity identification and risk assessment, dynamic and collaborative cross-chain governance, etc. Relevant skills/mathematical tools: BC crosschain technology, authentication protocols, smart contract programming.

## H. Integration with Large-scale Pre-trained Models

Large-scale pre-trained models hold the potential to offer exquisitely tailored and individualized encounters within the Metaverse. Through discerning analysis of user inclinations, behaviors, and historical data, these models can be refined to generate bespoke virtual panoramas [157], characters, and interactions, ushering in a realm of heightened immersion and personalized engagement within the Metaverse.

Nevertheless, notable challenges persist. One such challenge lies within the realm of the language macrocosm, where Metaverse users hail from diverse nations and regions, conversing in various languages. Consequently, it becomes imperative for the model to possess the capability to seamlessly interact across linguistic boundaries. Furthermore, the endeavor to ensure the seamless deployment of the model, endowed with high concurrency and unwavering stability, becomes an ineluctable quandary [158]. This ensures swift and responsive handling of users' requests, a matter of utmost significance. Relevant skills/mathematical tools: Natural language processing, deep learning model training, model deployment and optimization.

## I. Integration with Quantum Computing

Quantum computing, with its unparalleled capacity to handle intricate calculations and tackle problems exponentially faster, possesses the potential to reshape the capabilities of the Metaverse and elevate its performance to unprecedented heights. A recent study conducted by Cui et al. [159] delves into the implementation of secure communication and efficient cross-chain protocols utilizing quantum computing. Meanwhile, Ren et al. [160] have explored the utilization of gamebased quantum collective learning and many-to-many matching schemes within the Metaverse, enabling the attainment of optimal strategies for maximizing system revenue. Looking ahead, quantum computing is poised to find widespread application in enhancing data processing, encryption, and security performance within the Metaverse. Furthermore, it holds the promise of expediting virtual and AR experiences, thereby propelling the Metaverse into a realm of heightened immersion and seamless interactivity. Relevant skills/mathematical tools: Quantum computing theory, quantum algorithm design, quantum communication protocols.

## J. Availability of Datasets for Industrial Metaverse

In industrial environments, there is a plethora of data in different types, formats, and sources. These data often reside in disparate systems and devices, leading to data isolation and fragmentation. Lack of unified data integration and management mechanisms hampers seamless integration and consolidation of data. Moreover, various stakeholders in industrial environments may possess different data ownership and access rights. Establishing appropriate mechanisms for data ownership and access control is essential to ensure data legitimacy and compliance. Relevant skills/mathematical tools: Data management, data integration, access control.

## K. Ethical and Social Impacts

When machine decisions impact people's lives, especially in industrial scenarios with high cost and safety requirements, it is important to clarify how much autonomy the system allocates to machine algorithms. In addition to the technical aspects, fairness in resource allocation must also be considered [161]. For example, some large companies may find it easier to concentrate high-quality and extensive digital resources for DT modeling, while expensive hardware may exclude organizations with lower purchasing power, exacerbating the digital divide. It is necessary to enact more artificial intelligence laws in the future to subject the operation of the industrial Metaverse to legal jurisdiction.

## VI. CONCLUSION

In this paper, we conducted comprehensive research on the Industrial Metaverse. By analyzing the characteristics of Metaverse technology, we put forward convincing reasons for deploying Metaverse platforms in the industrial field. Furthermore, we summarized the research progress of several key enabling technologies in the Industrial Metaverse and elucidated the advantages of each technology in various stages of industrial production. We provided a comprehensive summary of various technologies, including existing challenges and performance requirements in industrial scenarios. Furthermore, starting from the Metaverse itself and combining with the characteristics of industrial scenarios, we identified three key challenges: security and confidentiality issues, resource allocation problems under resource constraints, and interoperability. In addition, we compiled the standardization efforts in the field of Metaverse over the past few years. Lastly, we concluded with a forward-looking discussion, considering the current challenges and future research directions. With the continuous advancement of Metaverse-related technologies and their further integration with industrial scenarios, we anticipate that our work can provide references for future research endeavors.

## REFERENCES

[1] Y. Zhao, J. Jiang, Y. Chen, R. Liu, Y. Yang, X. Xue, and S. Chen, "Metaverse: Perspectives from graphics, interactions and visualization," Vis. Informatics, vol. 6, pp. 56-67, Mar. 2022.

[2] K. Li, Y.-K. Cui, W. Li, T. Lv, X. Yuan, S. Li, W. Ni, M. Simsek, and F. Dressler, "When internet of things meets metaverse: Convergence of physical and cyber worlds," IEEE Internet Things J., vol. 10, pp. 4148-4173, Aug. 2022.

[3] B. Yang, S. Yang, Z. Lv, F. Wang, and T. Olofsson, "Application of digital twins and metaverse in the field of fluid machinery pumps and fans: A review," Sensors (Basel, Switzerland), vol. 22, Nov. 2022.

[4] L. Chang, Z. Zhang, P. Li, S. Xi, W.-X. Guo, Y. Shen, Z. Xiong, J. Kang, D. T. Niyato, X. Qiao, and Y. Wu, "6g-enabled edge ai for metaverse: Challenges, methods, and future research directions," ArXiv, vol. abs/2204.06192, 2022.

[5] P. Bhattacharya, D. Saraswat, D. Savaliya, S. Sanghavi, A. Verma, V. Sakariya, S. Tanwar, R. Sharma, M. S. Răboacă, and D. L. Manea, "Towards future internet: The metaverse perspective for diverse industrial applications," Mathematics, Feb. 2023.

[6] D. Mourtzis, J. D. Angelopoulos, and N. Panopoulos, "Blockchain integration in the era of industrial metaverse," Applied Sciences, Jan. 2023

[7] G. R. E. Said, "Metaverse-based learning opportunities and challenges: A phenomenological metaverse human-computer interaction study," Electronics, Mar. 2023.

[8] N. Kshetri, "The economics of the industrial metaverse," IT Prof., vol. 25, no. 1, pp. 84-88, 2023

[9] Z. Dong, X. Zhu, J. Cao, Y. Jiang, and V. K. N. Lau, "Task-oriented communications for industrial metaverse: Key techniques and open challenges," IEEE Internet Things Mag., vol. 6, no. 4, pp. 34-40, 2023.

[10] J. D. N. Dionisio, W. G. Burns, and R. Gilbert, "3d virtual worlds and the metaverse: Current status and future possibilities," ACM Comput. Surv., vol. 45, pp. 34:1-34:38, 2013.

[11] S.-M. Park and Y.-G. Kim, "A metaverse: Taxonomy, components, applications, and open challenges," IEEE Access, vol. 10, pp. 4209$4251,2022$.

[12] Y. Wang and J. Zhao, "Mobile edge computing, metaverse, $6 \mathrm{~g}$ wireless communications, artificial intelligence, and blockchain: Survey and their convergence," ArXiv, vol. abs/2209.14147, 2022.

[13] J. Yu, A. Y. Alhilal, P. Hui, and D. H. K. Tsang, " 6 g mobile-edge empowered metaverse: Requirements, technologies, challenges and research directions," ArXiv, vol. abs/2211.04854, 2022.

[14] B. Kye, N. Han, E. Kim, Y. Park, and S. Jo, "Educational applications of metaverse: possibilities and limitations," Journal of Educational Evaluation for Health Professions, vol. 18, 2021.
[15] A. Tlili, R. Huang, and Kinshuk, "Metaverse for climbing the ladder toward 'industry 5.0' and 'society 5.0'?" The Service Industries Journal, Feb. 2023.

[16] J. Li, Y. Shao, K. Wei, M. Ding, C. Ma, L. Shi, Z. Han, and H. V. Poor, "Blockchain assisted decentralized federated learning (BLADE-FL): Performance analysis and resource allocation," IEEE Trans. Parallel Distrib. Syst., vol. 33, no. 10, pp. 2401-2415, Dec. 2021.

[17] Y. Du, J. Li, L. Shi, Z. Wang, T. Wang, and Z. Han, "A novel oracleaided industrial iot blockchain: Architecture, challenges, and potential solutions," IEEE Netw., vol. 37, no. 3, pp. 8-15, 2023.

[18] Z. Lin, P. Xiangli, Z. Li, F. Liang, and A. Li, "Towards metaverse manufacturing: A blockchain-based trusted collaborative governance system," Shanghai, China, Mar. 2022

[19] H. Xu, Z. Li, Z. Li, X. Zhang, Y. Sun, and L. Zhang, "Metaverse native communication: A blockchain and spectrum prospective," Seoul, South Korea, Mar. 2022, pp. 7-12.

[20] J. Kang, D. Ye, J. Nie, J. Xiao, X. Deng, S. Wang, Z. Xiong, R. Yu, and D. T. Niyato, "Blockchain-based federated learning for industrial metaverses: Incentive scheme with optimal aoi," Espoo, Finland, Aug. 2022, pp. 71-78

[21] Z. Wang, Q. Hu, M. Xu, and H. Jiang, "Blockchain-based edge resource sharing for metaverse," Denver, Colorado, Aug. 2022, pp. 620-626.

[22] H. Huang, X. Zeng, L. Zhao, C. Qiu, H. Wu, and L. Fan, "Fusion of building information modeling and blockchain for metaverse: A survey," IEEE Open Journal of the Computer Society, vol. 3, pp. 195207, 2022.

[23] P. Bhattacharya, M. S. Obaidat, D. Savaliya, S. Sanghavi, S. Tanwar, and B. Sadoun, "Metaverse assisted telesurgery in healthcare 5.0: An interplay of blockchain and explainable ai," 2022 International Conference on Computer, Information and Telecommunication Systems (CITS), pp. 1-5, 2022

[24] R. Gupta, S. Tanwar, S. Tyagi, N. Kumar, M. S. Obaidat, and B. Sadoun, "Habits: Blockchain-based telesurgery framework for healthcare 4.0," 2019 International Conference on Computer, Information and Telecommunication Systems (CITS), pp. 1-5, Aug. 2019.

[25] A. R. Santhi and P. Muthuswamy, "Influence of blockchain technology in manufacturing supply chain and logistics," Logistics, Fed. 2022.

[26] X. Deng, J. Li, C. Ma, K. Wei, L. Shi, M. Ding, W. Chen, and H. V. Poor, "Blockchain assisted federated learning over wireless channels: Dynamic resource allocation and client scheduling," IEEE Trans. Wireless Commun., vol. 22, no. 5, pp. 3537-3553, May. 2021.

[27] Y. Wan, Y. Gao, and Y. Hu, "Blockchain application and collaborative innovation in the manufacturing industry: Based on the perspective of social trust," Technological Forecasting and Social Change, Apr. 2022

[28] X. Zhou, X. Xu, W. Liang, Z. Zeng, S. Shimizu, L. T. Yang, and Q. Jin, "Intelligent small object detection for digital twin in smart manufacturing with industrial cyber-physical systems," IEEE Transactions on Industrial Informatics, vol. 18, pp. 1377-1386, Feb. 2022.

[29] Y. Wu, H. Cao, G. Yang, T. Lu, and S. Wan, "Digital twin of intelligent small surface defect detection with cyber-manufacturing systems," ACM Transactions on Internet Technology, Nov. 2022.

[30] D. Guo, R. Y. Zhong, Y. Rong, and G. Q. Huang, "Synchronization of shop-floor logistics and manufacturing under iiot and digital twinenabled graduation intelligent manufacturing system," IEEE Transactions on Cybernetics, vol. 53, pp. 2005-2016, Sep. 2021.

[31] Z. Yin, Y. Lin, Y. Zhang, Y. Qian, F. Shu, and J. Li, "Collaborative multiagent reinforcement learning aided resource allocation for uav anti-jamming communication," IEEE Internet of Things Journal, vol. 9, no. 23, pp. 23 995-24 008, 2022.

[32] J. Yu, A. Y. Alhilal, P. Hui, and D. Tsang, "Bi-directional digital twin and edge computing in the metaverse," ArXiv, vol. abs/2211.08700, Nov. 2022

[33] C. Dai, K. Yang, and C. Deng, "A service placement algorithm based on merkle tree in mec systems assisted by digital twin networks," Chongqing, China, Dec. 2022, pp. 37-43.

[34] Y. Han, D. T. Niyato, C. Leung, D. I. Kim, K. Zhu, S. Feng, X. S Shen, and C. Miao, "A dynamic hierarchical framework for iot-assisted digital twin synchronization in the metaverse," IEEE Internet of Things Journal, vol. 10, pp. 268-284, Jan. 2023.

[35] J. Zhang, M. Zong, and W. Li, "A truthful mechanism for multibase station resource allocation in metaverse digital twin framework," Processes, Dec. 2022.

[36] D. V. Huynh, S. R. Khosravirad, A. Masaracchia, O. A. Dobre, and T. Q. Duong, "Edge intelligence-based ultra-reliable and low-latency communications for digital twin-enabled metaverse," IEEE Wireless Communications Letters, vol. 11, pp. 1733-1737, Aug. 2022.

[37] C. Yang, X. Tu, J. Autiosalo, R. Ala-Laurinaho, J. Mattila, P. Salminen, and K. Tammi, "Extended reality application framework for a digitaltwin-based smart crane," Applied Sciences, Jun. 2022.

[38] C. Coupry, S. Noblecourt, P. Richard, D. Baudry, and D. Bigaud, "Bimbased digital twin and $\mathrm{xr}$ devices to improve maintenance procedures in smart buildings: A literature review," Applied Sciences, Jul. 2021.

[39] S. Jerov and A. Tepljakov, "Digital twins in extended reality for control system applications," 2020 43rd International Conference on Telecommunications and Signal Processing (TSP), pp. 274-279, Jul. 2020.

[40] X. Tu, J. Autiosalo, R. Ala-Laurinaho, C. Yang, P. Salminen, and K. Tammi, "Twinxr: Method for using digital twin descriptions in industrial extended reality applications," in Frontiers in Virtual Reality, Jan. 2023.

[41] B. B. Gupta, A. Gaurav, K. T. Chui, L. Wang, V. Arya, A. Shukla, and D. Peraković, "Ddos attack detection through digital twin technique in metaverse," Las Vegas, NV, USA, Jan. 2023, pp. 1-5.

[42] L. Zhang, L. Feng, J. Wang, and K.-S. Lin, "Integration of design, manufacturing, and service based on digital twin to realize intelligent manufacturing," Machines, Apr. 2022.

[43] I. Onaji, D. Tiwari, P. Soulatiantork, B. Song, and A. Tiwari, "Digital twin in manufacturing: conceptual framework and case studies," International Journal of Computer Integrated Manufacturing, vol. 35, pp. 831 - 858, Jan. 2022.

[44] D. Mourtzis, "Digital twin inception in the era of industrial metaverse," in Frontiers in Manufacturing Technology, Apr. 2023.

[45] B. Siniarski, C. de Alwis, G. Yenduri, T. Huynh-The, G. Gür, T. R. Gadekallu, and M. Liyanage, "Need of $6 \mathrm{~g}$ for the metaverse realization," ArXiv, vol. abs/2301.03386, Dec. 2022.

[46] J. Huang, H. Gao, S. Wan, and Y. Chen, "Aoi-aware energy control and computation offloading for industrial iot," Future Gener. Comput. Syst., vol. 139, pp. 29-37, Sep. 2022.

[47] J. Cao, X. Zhu, S. Sun, Z. Wei, Y. Jiang, J. Wang, and V. K. N. Lau, "Toward industrial metaverse: Age of information, latency and reliability of short-packet transmission in 6g," IEEE Wireless Communications, vol. 30, pp. 40-47, Apr. 2023.

[48] A. M. Aslam, R. Chaudhary, A. Bhardwaj, I. Budhiraja, N. Kumar, and S. Zeadally, "Metaverse for $6 \mathrm{~g}$ and beyond: The next revolution and deployment challenges," IEEE Internet of Things Magazine, vol. 6, pp. 32-39, Mar. 2023.

[49] P. K. Padhi and F. Charrua-Santos, " $6 \mathrm{~g}$ enabled industrial internet of everything: Towards a theoretical framework," Applied System Innovation, Feb. 2021.

[50] H. Jalo, H. Pirkkalainen, O. Torro, E. Pessot, A. Zangiacomi, and A. Tepljakov, "Extended reality technologies in small and mediumsized european industrial companies: level of awareness, diffusion and enablers of adoption," Virtual Reality, vol. 26, pp. 1745 - 1761, Jun. 2022.

[51] J. Guan, J. Irizawa, and A. Morris, "Extended reality and internet of things for hyper-connected metaverse environments," Christchurch, New Zealand, Mar. 2022, pp. 163-168.

[52] T. Holz, A. G. Campbell, G. M. P. O'Hare, J. W. Stafford, A. N. Martin, and M. Dragone, "Mira - mixed reality agents," Int. J. Hum. Comput. Stud., vol. 69, pp. 251-268, Apr. 2011.

[53] J. A. Rincon, J.-L. Poza-Luján, V. Julián, J.-L. Posadas-Yagüe, and C. Carrascosa, "Extending mam5 meta-model and jacaliv e framework to integrate smart devices from real environments," PLoS ONE, vol. 11, Feb. 2016.

[54] A. Croatti and A. Ricci, "A model and platform for building agentbased pervasive mixed reality systems," in Practical Applications of Agents and Multi-Agent Systems, Jun. 2018.

[55] R. Prada, I. S. W. B. Prasetya, F. M. Kifetew, F. Dignum, T. E. J. Vos, J. Lander, J.-Y. Donnart, A. Kazmierowski, J. Davidson, and P. M. Fernandes, "Agent-based testing of extended reality systems," Porto, Portugal, Oct. 2020, pp. 414-417.

[56] T. Braud, L.-H. Lee, A. Y. Alhilal, C. B. Fernandez, and P. Hui, "Dios - an extended reality operating system for the metaverse," ArXiv, vol. abs/2201.03256, Jan. 2022.

[57] Y. Shao, N. Lessio, and A. Morris, "Iot avatars: Mixed reality hybrid objects for core ambient intelligent environments," Procedia Computer Science, 2019.

[58] A. Dünser and M. Billinghurst, "Evaluating augmented reality systems," in Handbook of Augmented Reality, 2011.

[59] T. Tsang and A. Morris, "A hybrid quality-of-experience taxonomy for mixed reality iot (xri) systems," Melbourne, Australia, Apr. 2021, pp. $1809-1816$.
[60] A. Morris, J. Guan, and A. Azhar, "An xri mixed-reality internet-ofthings architectural framework toward immersive and adaptive smart environments," Bari, Italy, Oct. 2021, pp. 68-74.

[61] L. Oppermann, F. P. Buchholz, and Y. Uzun, "Industrial metaverse: Supporting remote maintenance with avatars and digital twins in collaborative xr environments," Extended Abstracts of the 2023 CHI Conference on Human Factors in Computing Systems, Apr. 2023.

[62] B. H. Kantowitz, "Mental workload," Encyclopedia of Behavioral Medicine, 2020.

[63] L.-H. Lee, T. Braud, P. Zhou, L. Wang, D. Xu, Z. Lin, A. Kumar, C. Bermejo, and P. Hui, "All one needs to know about metaverse: A complete survey on technological singularity, virtual ecosystem, and research agenda," ArXiv, vol. abs/2110.05352, 2021.

[64] T. J. Chua, W. li Yu, and J. Zhao, "Resource allocation for mobile metaverse with the internet of vehicles over $6 \mathrm{~g}$ wireless communications: A deep reinforcement learning approach," ArXiv, vol. abs/2209.13425, 2022.

[65] H. Kwon, K. Nair, J. Seo, J. Yik, D. Mohapatra, D. Zhan, J. Song, P. L. Capak, P. Zhang, P. Vajda, C. R. Banbury, M. Mazumder, L. Lai, A. Sirasao, T. Krishna, H. Khaitan, V. Chandra, and V. J. Reddi, "Xrbench: An extended reality (xr) machine learning benchmark suite for the metaverse," ArXiv, vol. abs/2211.08675, 2022.

[66] C. Warin and D. Reinhardt, "Vision: Usable privacy for xr in the era of the metaverse," Proceedings of the 2022 European Symposium on Usable Security, 2022.

[67] N. Xi, J. Chen, F. Gama, M. Riar, and J. Hamari, "The challenges of entering the metaverse: An experiment on the effect of extended reality on workload," Information Systems Frontiers, pp. 1 - 22, 2022.

[68] H. Zhu, "Metaaid: A flexible framework for developing metaverse applications via ai technology and human editing," ArXiv, vol. abs/2204.01614, 2022.

[69] S. M. Wong, C.-W. Chen, T.-Y. Pan, H.-K. Chu, and M.-C. Hu, "Getwild: A vr editing system with ai-generated 3d object and terrain," Proceedings of the 30th ACM International Conference on Multimedia, Oct. 2022.

[70] L. Yang, R. M. Radway, Y.-H. Chen, T. F. Wu, H. Liu, E. Ansari, V. Chandra, S. Mitra, and E. Beigné, "Three-dimensional stacked neural network accelerator architectures for ar/vr applications," IEEE Micro, vol. 42, pp. 116-124, Nov. 2022.

[71] S. Badruddoja, R. Dantu, Y. He, M. A. Thompson, A. Salau, and K. Upadhyay, "Trusted ai with blockchain to empower metaverse," 2022 Fourth International Conference on Blockchain Computing and Applications (BCCA), pp. 237-244, 2022.

[72] O. Khalaj, M. B. Jamshidi, P. Hassas, M. Hosseininezhad, B. Maek, C. tadler, and J. Svoboda, "Metaverse and ai digital twinning of 42sicr steel alloys," Mathematics, 2022.

[73] C. Ma, J. L. K. Wei, B. Liu, M. Ding, L. Yuan, Z. Han, and H. V. Poor, "Trusted AI in multi-agent systems: An overview of privacy and security for distributed learning," CoRR, vol. abs/2202.09027, 2022.

[74] J. Yang, T. Liu, B. Jiang, H. Song, and W. Lu, "3d panoramic virtual reality video quality assessment based on $3 \mathrm{~d}$ convolutional neural networks," IEEE Access, vol. 6, pp. 38 669-38682, 2018.

[75] C.-H. Yeh, C.-H. Huang, and L.-W. Kang, "Multi-scale deep residual learning-based single image haze removal via image decomposition," IEEE Transactions on Image Processing, vol. 29, pp. 3153-3167, Dec. 2019 .

[76] S. Satrasupalli, E. Daniel, and S. R. Guntur, "Single image haze removal based on transmission map estimation using encoder-decoder based deep learning architecture," Optik, Dec. 2021.

[77] Q. Liu, L. Shi, L. Sun, J. Li, M. Ding, and F. Shu, "Path planning for uav-mounted mobile edge computing with deep reinforcement learning," IEEE Trans. Veh. Technol., vol. 69, no. 5, pp. 5723-5728, Jan. 2020 .

[78] J. Lee, A. Pastor, J.-I. Hwang, and G. J. Kim, "Predicting the torso direction from hmd movements for walk-in-place navigation through deep learning," Proceedings of the 25th ACM Symposium on Virtual Reality Software and Technology, Dec. 2019

[79] H. Du, Z. Li, D. T. Niyato, J. Kang, Z. Xiong, H. Huang, and S. Mao, "Generative ai-aided optimization for ai-generated content (aigc) services in edge networks," ArXiv, vol. abs/2303.13052, Mar. 2023.

[80] M. Xu, D. T. Niyato, H. Zhang, J. Kang, Z. Xiong, S. Mao, and Z. Han, "Sparks of gpts in edge intelligence for metaverse: Caching and inference for mobile aigc services," ArXiv, vol. abs/2304.08782, Apr. 2023.

[81] L.-H. Lee, P. Zhou, C. Zhang, and S. J. Hosio, "What if we have meta gpt? from content singularity to human-metaverse interaction in aigc era," ArXiv, vol. abs/2304.07521, Apr. 2023.

[82] B. K. Wiederhold, "Treading carefully in the metaverse: The evolution of ai avatars," Cyberpsychology, behavior and social networking, vol. 265 , pp. 321-322, Jan. 2023.

[83] S. Mohseni, N. Zarei, and E. D. Ragan, "A multidisciplinary survey and framework for design and evaluation of explainable ai systems," ACM Trans. Interact. Intell. Syst., vol. 11, pp. 24:1-24:45, 2018.

[84] J. Han and Y. Lee, "Explainable artificial intelligence-based competitive factor identification," ACM Transactions on Knowledge Discovery from Data (TKDD), vol. 16, pp. $1-11,2021$.

[85] M. Xu, W. C. Ng, W. Y. B. Lim, J. Kang, Z. Xiong, D. T. Niyato, Q. Yang, X. S. Shen, and C. Miao, "A full dive into realizing the edgeenabled metaverse: Visions, enabling technologies, and challenges," ArXiv, vol. abs/2203.05471, 2022.

[86] X. Zhai, X. Chu, M. Wang, Z. Zhang, and Y. Dong, "Education metaverse: Innovations and challenges of the new generation of internet education formats," Metaverse, May. 2022.

[87] S. E. Bibri and Z. Allam, "The metaverse as a virtual form of datadriven smart cities: the ethics of the hyper-connectivity, datafication, algorithmization, and platformization of urban society," Computational Urban Science, vol. 2, Jun. 2022.

[88] J. N. Njoku, C. I. Nwakanma, G. C. Amaizu, and D. Kim, "Prospects and challenges of metaverse application in data-driven intelligent transportation systems," IET Intelligent Transport Systems, Aug. 2022.

[89] H. Wu, X. Tian, Y. Gong, X. Su, M. Li, and F. Xu, "Dapter: Preventing user data abuse in deep learning inference services," Singapore, Singapore, Jun. 2021.

[90] R. Zhao, Y. Zhang, Y. Zhu, R. Lan, and Z. Hua, "Metaverse: Security and privacy concerns," ArXiv, vol. abs/2203.03854, Mar. 2022.

[91] Z.-H. Pang, L.-Z. Fan, Z. Dong, Q.-L. Han, and G. Liu, "False data injection attacks against partial sensor measurements of networked control systems," IEEE Transactions on Circuits and Systems II: Express Briefs, vol. 69, pp. 149-153, Jan. 2022.

[92] H. Q. Guo, Z. Pang, J. Sun, and J. Li, "Detection of stealthy false data injection attacks against cyber-physical systems: A stochastic coding scheme," Journal of Systems Science and Complexity, vol. 35, pp. 1668 - 1684, Aug. 2022.

[93] B. Bera, A. K. Das, M. S. Obaidat, P. Vijayakumar, K.-F. Hsiao, and Y. Park, "Ai-enabled blockchain-based access control for malicious attacks detection and mitigation in ioe," IEEE Consumer Electronics Magazine, vol. 10, pp. 82-92, Sep. 2021.

[94] S. Ruj, M. Stojmenovic, and A. Nayak, "Decentralized access control with anonymous authentication of data stored in clouds," IEEE Transactions on Parallel and Distributed Systems, vol. 25, pp. 384-394, Feb. 2014.

[95] M. Shen, A. Liu, G. Huang, N. N. Xiong, and H. Lu, "Attdc: An active and traceable trust data collection scheme for industrial security in smart cities," IEEE Internet of Things Journal, vol. 8, pp. 64376453, Apr. 2021.

[96] A. M. Palacios, M. J. Gacto, and J. Alcalá-Fdez, "Mining fuzzy association rules from low-quality data," Soft Computing, vol. 16, pp. 883-901, May. 2012.

[97] R. Duan, J. Liu, J. Zhou, P. Wang, and W. Liu, "An ensemble prognostic method of francis turbine units using low-quality data under variable operating conditions," Sensors (Basel, Switzerland), vol. 22, Italian, Jan. 2022.

[98] B. Yang, Z. Liu, G. Duan, and J. Tan, "Mask2defect: A prior knowledge-based data augmentation method for metal surface defect inspection," IEEE Transactions on Industrial Informatics, vol. 18, pp. 6743-6755, Oct. 2022

[99] S. Suhail, R. Hussain, R. Jurdak, A. Oracevic, K. Salah, C. S. Hong, and R. Matuleviius, "Blockchain-based digital twins: Research trends, issues, and future challenges," ACM Computing Surveys (CSUR), vol. 54, pp. $1-34$, Mar. 2021.

[100] K. Peng, H. Huang, M. Bilal, and X. Xu, "Distributed incentives for intelligent offloading and resource allocation in digital twin driven smart industry," IEEE Transactions on Industrial Informatics, vol. 19, pp. 3133-3143, Mar. 2023.

[101] W. Shen, T. Hu, C. Zhang, and S. Ma, "Secure sharing of big digital twin data for smart manufacturing based on blockchain," Journal of Manufacturing Systems, Oct. 2021.

[102] Y. Wang, Z. Su, N. Zhang, R. Xing, D. Liu, T. H. Luan, and X. Shen, "A survey on metaverse: Fundamentals, security, and privacy," IEEE Communications Surveys \& Tutorials, vol. 25, no. 1, pp. 319-352, 2023.
[103] S. Kim, W. Jo, and T. Shon, "Apad: Autoencoder-based payload anomaly detection for industrial ioe," Appl. Soft Comput., vol. 88, p. 106017, Mar. 2020.

[104] S. Kwon, H. Yoo, and T. Shon, "Ieee 1815.1-based power system security with bidirectional rnn-based network anomalous attack detection for cyber-physical system," IEEE Access, vol. 8, pp. 77 572-77 586, Apr. 2020.

[105] Y. Wang, Z. Su, N. Zhang, D. Liu, R. Xing, T. H. Luan, and X. S. Shen, "A survey on metaverse: Fundamentals, security, and privacy," ArXiv, vol. abs/2203.02662, 2022.

[106] X. He, Q. Gong, Y. Chen, Y. Zhang, X. Wang, and X. Fu, "Datingsec: Detecting malicious accounts in dating apps using a content-based attention network," IEEE Transactions on Dependable and Secure Computing, vol. 18, pp. 2193-2208, Sep. 2021.

[107] S. Corbett-Davies, E. Pierson, A. Feller, S. Goel, and A. Z. Huq, "Algorithmic decision making and the cost of fairness," Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, Jan. 2017.

[108] A. Datta, S. Sen, and Y. Zick, "Algorithmic transparency via quantitative input influence: Theory and experiments with learning systems," California, USA, May. 2016, pp. 598-617.

[109] S. Cresci, M. Petrocchi, A. Spognardi, and S. Tognazzi, "Adversarial machine learning for protecting against online manipulation," IEEE Internet Computing, vol. 26, pp. 47-52, Nov. 2021.

[110] D. Mingxiao, M. Xiaofeng, Z. Zhe, W. Xiangwei, and C. Qijun, "A review on consensus algorithm of blockchain," 2017 IEEE International Conference on Systems, Man, and Cybernetics (SMC), pp. 2567-2572, Oct. 2017.

[111] L. Wang, W. Chen, N. Jing, Z. Chang, B. Li, and W. Liu, "Acopalm: Acoustical palmprint-based noncontact identity authentication," IEEE Transactions on Industrial Informatics, vol. 18, pp. 9122-9131, Dec. 2022.

[112] R. Duezguen, P. Mayer, S. Das, and M. Volkamer, "Towards secure and usable authentication for augmented and virtual reality head-mounted displays," ArXiv, vol. abs/2007.11663, Jul. 2020.

[113] X. Deng, J. Li, C. Ma, K. Wei, L. Shi, M. Ding, and W. Chen, "Lowlatency federated learning with dnn partition in distributed industrial iot networks," IEEE J. Sel. Areas Commun., vol. 41, no. 3, pp. 755-775, Oct. 2022.

[114] Z. Yin, Z. Wang, J. Li, M. Ding, W. Chen, and S. Jin, "Decentralized federated reinforcement learning for user-centric dynamic tfdd control," IEEE Journal of Selected Topics in Signal Processing, vol. 17, no. 1, pp. 40-53, 2023.

[115] M. Al-Quraan, L. S. Mohjazi, L. Bariah, A. Centeno, A. Zoha, K. Arshad, K. Assaleh, S. H. Muhaidat, M. Debbah, and M. A. Imran, "Edge-native intelligence for $6 \mathrm{~g}$ communications driven by federated learning: A survey of trends and challenges," IEEE Transactions on Emerging Topics in Computational Intelligence, vol. 7, pp. 957-979, Nov. 2021.

[116] Y. Chen, S. Huang, W. Gan, G. Huang, and Y. Wu, "Federated learning for metaverse: A survey," Companion Proceedings of the ACM Web Conference 2023, 2023.

[117] W. Zhang, G. Zhang, and S. Mao, "Joint parallel offloading and load balancing for cooperative-mec systems with delay constraints," IEEE Trans. Veh. Technol., vol. 71, no. 4, pp. 4249-4263, 2022.

[118] B. Tan, L. Ai, M. Wang, and J. Wang, "Toward A task offloading framework based on cyber digital twins in mobile edge computing," IEEE Wirel. Commun., vol. 30, no. 3, pp. 157-162, 2023.

[119] W. C. Ng, W. Y. B. Lim, J. S. Ng, Z. Xiong, D. Niyato, and C. Miao, "Unified resource allocation framework for the edge intelligenceenabled metaverse," in IEEE International Conference on Communications, ICC 2022, Seoul, Korea, May 16-20, 2022. IEEE, 2022, pp. $5214-5219$.

[120] N. H. Chu, D. N. Nguyen, D. T. Hoang, K. T. Phan, E. Dutkiewicz, D. Niyato, and T. Shu, "Dynamic resource allocation for metaverse applications with deep reinforcement learning," in IEEE Wireless Communications and Networking Conference, WCNC 2023, Glasgow, UK, March 26-29, 2023. IEEE, 2023, pp. 1-6.

[121] P. Yu, M. Yang, A. Xiong, Y. Ding, W. Li, X. Qiu, L. Meng, M. Kadoch, and M. Cheriet, "Intelligent-driven green resource allocation for industrial internet of things in $5 \mathrm{~g}$ heterogeneous networks," IEEE Trans. Ind. Informatics, vol. 18, no. 1, pp. 520-530, 2022.

[122] N. C. Luong, Q.-V. Pham, T. Huynh-The, V.-D. Nguyen, D. W. K. $\mathrm{Ng}$, and S. Chatzinotas, "Edge computing for semantic communication enabled metaverse: An incentive mechanism design," ArXiv, vol. $\mathrm{abs} / 2212.06463$, Dec. 2022.

[123] M. Li, J. Weng, A. Yang, W. Lu, Y. Zhang, L. Hou, J.-N. Liu, Y. Xiang, and R. H. Deng, "Crowdbc: A blockchain-based decentralized framework for crowdsourcing," IEEE Transactions on Parallel and Distributed Systems, vol. 30, pp. 1251-1266, Jun. 2019.

[124] Y. Han, D. T. Niyato, C. Leung, C. Miao, and D. I. Kim, "A dynamic resource allocation framework for synchronizing metaverse with iot service and data," Oct. 2021, pp. 1196-1201.

[125] M. T. Rashid, D. Zhang, and D. Wang, "Edgestore: Towards an edge-based distributed storage system for emergency response," in HPCC/SmartCity/DSS. IEEE, 2019, pp. 2543-2550.

[126] K. Poularakis, J. Llorca, A. M. Tulino, I. J. Taylor, and L. Tassiulas, "Joint service placement and request routing in multi-cell mobile edge computing networks," in INFOCOM. IEEE, 2019, pp. 10-18.

[127] P. Liu, G. Xu, K. Yang, K. Wang, and X. Meng, "Jointly optimized energy-minimal resource allocation in cache-enhanced mobile edge computing systems," IEEE Access, vol. 7, pp. 3336-3347, 2019.

[128] D. Zhang, Y. Ma, Y. Zhang, S. Lin, X. S. Hu, and D. Wang, "A realtime and non-cooperative task allocation framework for social sensing applications in edge computing systems," in RTAS. IEEE Computer Society, 2018, pp. 316-326.

[129] L. M. Vaquero, F. Cuadrado, Y. Elkhatib, J. B. Bernabé, S. N. Srirama, and M. F. Zhani, "Research challenges in nextgen service orchestration," Future Gener. Comput. Syst., vol. 90, pp. 20-38, 2019.

[130] Y. Cai, J. Llorca, A. M. Tulino, and A. F. Molisch, "Joint computecaching-communication control for online data-intensive service delivery," CoRR, vol. abs/2205.01944, 2022.

[131] D. Van-Huynh, S. R. Khosravirad, A. Masaracchia, O. A. Dobre, and T. Q. Duong, "Edge intelligence-based ultra-reliable and lowlatency communications for digital twin-enabled metaverse," IEEE Wirel. Commun. Lett., vol. 11, no. 8, pp. 1733-1737, 2022.

[132] P. A. Rospigliosi, "Metaverse or simulacra? roblox, minecraft, meta and the turn to virtual reality for education, socialisation and work," Interactive Learning Environments, vol. 30, pp. 1 - 3, Jan. 2022.

[133] T. Li, C. Yang, Q. Yang, S. Zhou, H. Huang, and Z. Zheng, "Metaopera: A cross-metaverse interoperability protocol," CoRR, vol. $\mathrm{abs} / 2302.01600,2023$.

[134] T. Huynh-The, T. R. Gadekallu, W. Wang, G. Yenduri, P. Ranaweera, Q. Pham, D. B. da Costa, and M. Liyanage, "Blockchain for the metaverse: A review," Future Gener. Comput. Syst., vol. 143, pp. 401419, 2023.

[135] 2019 IEEE Symposium on Security and Privacy, SP 2019, San Francisco, CA, USA, May 19-23, 2019. IEEE, 2019.

[136] S. Ghirmai, D. Mebrahtom, M. Aloqaily, M. Guizani, and M. Debbah, "Self-sovereign identity for trust and interoperability in the metaverse," CoRR, vol. abs/2303.00422, 2023.

[137] F. Khan, R. L. Kumar, M. H. Abidi, S. Kadry, H. Alkhalefah, and M. K. Aboudaif, "Federated split learning model for industry 5.0: A data poisoning defense for edge computing," Electronics, 2022.

[138] 23rd IEEE International Conference on Information Reuse and Integration for Data Science, IRI 2022, San Diego, CA, USA, August 9-11, 2022. IEEE, 2022

[139] O. Hashash, C. Chaccour, W. Saad, K. Sakaguchi, and T. Yu, "Towards a decentralized metaverse: Synchronized orchestration of digital twins and sub-metaverses," CoRR, vol. abs/2211.14686, 2022.

[140] U. Jaimini, T. Zhang, G. O. Brikis, and A. P. Sheth, "imetaversekg: Industrial metaverse knowledge graph to promote interoperability in design and engineering applications," IEEE Internet Comput., vol. 26, no. 6, pp. 59-67, 2022.

[141] T. Li, C.-M. Yang, Q. Yang, S. Zhou, H. Huang, and Z. Zheng, "Metaopera: A cross-metaverse interoperability protocol," ArXiv, vol. $\mathrm{abs} / 2302.01600$, Feb. 2023.

[142] B. S. Rawal, A. Mentges, and S. Ahmad, "The rise of metaverse and interoperability with split-protocol," San Diego, CA, USA, Aug. 2022, pp. 192-199.

[143] 25th International Conference on Advanced Communication Technology, ICACT 2023, Pyeongchang, Korea, Republic of, February 19-22, 2023. IEEE, 2023.

[144] J.-S. Choi and H.-G. Byun, "A case study to standardize odor metadata obtained from coffee aroma based on e-nose using iso/iec 23005 (mpeg-v) for olfactory-enhanced multimedia," JOURNAL OF SENSOR SCIENCE AND TECHNOLOGY, 2021.

[145] K. Yoon, S.-K. Kim, S. Jeong, and J.-H. Choi, "Interfacing cyber and physical worlds: Introduction to ieee 2888 standards," 2021 IEEE International Conference on Intelligent Reality (ICIR), pp. 49-50, May. 2021.

[146] "Ieee draft standard for general requirements for identity framework for metaverse," IEEE P3812.1/D1.1, March 2023, pp. 1-23, 2023.
[147] "Ieee draft standard for augmented reality on mobile devices: General requirements for software framework, components, and integration," IEEE P2048.101/D4.0, January 2023, pp. 1-34, 2023.

[148] T. Braud, C. B. Fernandez, and P. Hui, "Scaling-up ar: University campus as a physical-digital metaverse," Christchurch, New Zealand, Mar. 2022, pp. 169-175.

[149] T. Huynh-The, Q.-V. Pham, X.-Q. Pham, T. T. Nguyen, Z. Han, and D.-S. Kim, "Artificial intelligence for the metaverse: A survey," Eng. Appl. Artif. Intell., vol. 117, p. 105581, Feb. 2022.

[150] A. M. Aburbeian, A. Y. Owda, and M. Owda, "A technology acceptance model survey of the metaverse prospects," AI, Apr. 2022.

[151] S. Park and S. Kim, "Identifying world types to deliver gameful experiences for sustainable learning in the metaverse," Sustainability, Jan. 2022 .

[152] W. Suh and S. Ahn, "Utilizing the metaverse for learner-centered constructivist education in the post-pandemic era: An analysis of elementary school students," Journal of Intelligence, vol. 10, Mar. 2022.

[153] M. Golf-Pape, J. Heller, T. Hilken, M. B. Chylinski, K. de Ruyter, D. I. Keeling, and D. Mahr, "Embracing falsity through the metaverse: The case of synthetic customer experiences," Business Horizons, 2022.

[154] D.-I. D. Han, Y. Bergs, and N. Moorhouse, "Virtual reality consumer experience escapes: preparing for the metaverse," Virtual Reality, vol. 26, pp. 1443 - 1458, 2022.

[155] Y. Shao, J. Li, M. Ding, K. Wei, C. Ma, L. Shi, W. Chen, and S. Jin, "Design of anti-plagiarism mechanisms in decentralized federated learning," IEEE Trans. Serv. Comput., Early Access, Mar. 2024.

[156] X. Zhang, Y. Chen, L. Hu, and Y. Wang, "The metaverse in education: Definition, framework, features, potential applications, challenges, and future research topics," Frontiers in Psychology, vol. 13, 2022.

[157] S. J. Kwon, J. Kim, J. Bae, K. M. Yoo, J.-H. Kim, B. Park, B. Kim, J.-W. Ha, N. Sung, and D. Lee, "Alphatuning: Quantization-aware parameter-efficient adaptation of large-scale pre-trained language models," ArXiv, vol. abs/2210.03858, Oct. 2022.

[158] N. Ding, Y. Qin, G. Yang, F. Wei, Z. Yang, Y. Su, S. Hu, Y. Chen, C.-M. Chan, W. Chen, J. Yi, W. Zhao, X. Wang, Z. Liu, H. Zheng, J. Chen, Y. Liu, J. Tang, J. Li, and M. Sun, "Parameter-efficient finetuning of large-scale pre-trained language models," Nature Machine Intelligence, vol. 5, pp. 220-235, Mar. 2023.

[159] Y. Cui, "A cross-chain protocol based on quantum teleportation for underlying architecture of metaverse," Wuhan, China, Apr. 2022, pp. $508-512$.

[160] Y. Ren, R. Xie, F. Yu, T. Huang, and Y. jie Liu, "Quantum collective learning and many-to-many matching game in the metaverse for connected and autonomous vehicles," IEEE Transactions on Vehicular Technology, vol. 71, pp. 12 128-12 139, Nov. 2022.

[161] R. Benjamins, Y. R. Viñuela, and C. Alonso, "Social and ethical challenges of the metaverse," AI Ethics, vol. 3, no. 3, pp. 689-697, 2023 .

</end of paper 3>


