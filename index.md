<add in camera culture logo>
    
# Split Learning : Distributed deep learning without sharing raw data #
<p align="center"><a href=""><img src="https://splitlearning.github.io/diab1.png" height="320" width="600"></a></p>

Split learning naturally allows for various configurations of cooperating entities to train (and infer from) machine learning  models without sharing any raw data or detailed information about the model. This method has been developed by the MIT Media Lab’s Camera Culture group.

<h4> Team </h4>
    Ramesh Raskar, Associate Professor, MIT Media Lab; Project Director (raskar(at)mit.edu)<br>
    Praneeth Vepakomma,Research Assitant, MIT Media Lab<br>
<h5> Current Collaborations </h5>
    MGH <br>
    Martinos Center<br>

<a href="http://www.youtube.com/watch?feature=player_embedded&v=GiGlHuWOwME
" target="_blank"><img src="http://img.youtube.com/vi/GiGlHuWOwME/0.jpg" 
alt="Health Grid: Blockchain-based Data Marketplace | Ramesh Raskar | WEF 2019" width="240" height="180" /></a>     <a href="http://www.youtube.com/watch?feature=player_embedded&v=8GtJ1bWHZvg
" target="_blank"><img src="http://img.youtube.com/vi/8GtJ1bWHZvg/0.jpg" 
alt="RAMESH RASKAR INTERVIEW WITH BLOXLIVE AT THE WEF" width="240" height="180" /></a> <a href="http://www.youtube.com/watch?feature=player_embedded&v=7jWXaABY81I
" target="_blank"><img src="http://img.youtube.com/vi/7jWXaABY81I/0.jpg" 
alt="AI for All | Speedtalk | Ramesh Raskar" width="240" height="180" /></a>
<br /><br />

**Abstract:**  In the simplest of configurations of split learning, each client (for example, radiology center) trains a partial deep  network up to a specific layer known as the cut layer. The outputs at the cut layer are sent to another entity  (server/another client) which completes the rest of the training without looking at raw data from
any client that holds the raw data. This completes a round of forward propagation without sharing raw data. The gradients
are now back propagated again from its last layer until the cut layer in a similar fashion. The gradients at the
cut layer (and only these gradients) are sent back to radiology client centers. The rest of back
propagation is now completed at the radiology client centers. This process is continued until the
distributed split learning network is trained without looking at each others raw data.</font>

## Outline
- [Frequently asked questions](#faq)
- [Upcoming tutorial](#ut)
- [Further Reading](#fr)
- [Efficiency and plug-and-play configurations of split learning](#sl)
- [News](#news)

<h3 id="faq"> Frequently asked questions </h3>
1. <strong>How does split learning work and what is new in our approach? </strong><br/>
    Split learning attains high resource efficiency for distributed deep learning in comparison to existing methods by splitting the models architecture across distributed entities. It only communicates activations and gradients just from the split layer unlike other popular methods that share weights/gradients from all the layers. Split learning requires no raw data sharing; either of labels or features.<br><br>

2. <strong> How is raw data protected and who can get positively impacted? </strong><br/>
    Split learning requires absolutely no raw data sharing. Sectors like healthcare, finance, security, surveillance and others where data sharing is prohibited will benefit from our approach for training distributed deep learning models. Another modality of split learning called NoPeek SplitNN also drastically reduces leakage due to any communicated activations by reducing their distance correlation with raw data while maintaining model performance via categorical cross-entropy.<br /><br/>

3.<strong> How long will it take to transition from laboratory setting to actual deployment between cooperating entities?<br/></strong>
The approach is easily deployable for inter and intra entity or organizational collaboration and is highly versatile in terms of possible network topologies. Due to its high resource efficiency in terms of computations, memory, communication bandwidth it is also naturally suitable for distributed learning where the clients are pervasive and ubiquitous edge devices like mobile phones or IOT devices as well as across larger devices and organizations. 
<br /><br />
    
4. How is privacy maintained and who can get impacted?
    • Talk about application scenarios
    

<h3 id="fr"> Further reading </h3>
<h4> Split Learning Papers </h4>
1. Reducing leakage in distributed deep learning for sensitive health data, Praneeth Vepakomma, Otkrist Gupta, Abhimanyu Dubey, Ramesh Raskar, Accepted to ICLR 2019 Workshop on AI for social good. (2019)
2. Distributed learning of deep neural network over multiple agents, Otkrist Gupta and Ramesh Raskar, In: Journal of Network and Computer Applications 116, [(PDF)](https://www.sciencedirect.com/science/article/pii/S1084804518301590 "Pdf") (2018)
3. Split learning for health: Distributed deep learning without sharing raw patient data, Praneeth Vepakomma, Otkrist Gupta, Tristan Swedish, Ramesh Raskar, [(PDF)](https://arxiv.org/pdf/1812.00564.pdf "Pdf") (2018)
4. Survey paper: No Peek: A Survey of private distributed deep learning, Praneeth Vepakomma, Tristan Swedish, Ramesh Raskar, Otkrist Gupta, Abhimanyu Dubey, [(PDF)](https://arxiv.org/pdf/1812.03288.pdf "Pdf") (2018)

<h4> AutoML Papers </h4>
1. Accelerating neural architecture search using performance prediction, Bowen Baker, Otkrist Gupta, Ramesh Raskar, Nikhil Naik, In: conference paper at ICLR, [(PDF)](https://arxiv.org/pdf/1705.10823.pdf "Pdf") (2018)
2. Designing neural network architecture using reinforcement learning, Bowen Baker, Otkrist Gupta, Nikhil Naik & Ramesh Raskar, In: conference paper at ICLR, [(PDF)](https://arxiv.org/pdf/1611.02167.pdf "Pdf") (2017)

<h3>  Slides on split learning for data transparent ML </h3> 
<p align="center"><a href="https://www.slideshare.net/cameraculture/split-learning-versus-federated-learning-for-data-transparent-ml"><img src="https://splitlearning.github.io/splitSlides.png" height="250" width="300" ></a></p>
<br /><br />

<h3 id="ut"> Upcoming: CVPR Tutorial on “Distributed Private Machine Learning for Computer Vision: Federated Learning and Beyond”</h3>
We are giving a half-day tutorial at CVPR 2019: 
On Distributed Private Machine Learning for Computer Vision: Federated Learning, Split Learning and Beyond by
<b> Brendan McMahan (Google, USA)</b>, <b>Jakub Konečný</b> (Google, USA), <b>Otkrist Gupta (LendBuzz)</b>, <b>Ramesh Raskar</b> (MIT Media Lab, Cambridge, Massachusetts, USA),<b> Hassan Takabi</b> (University of North Texas, Texas, USA) and <b>Praneeth Vepakomma</b> (MIT Media Lab, Cambridge, Massachusetts, USA).
<br /><br />

<h3 id="sl"> Efficiency and plug-and-play configurations of split learning </h3>
<h4> Split learning's computational and communication efficiency on clients </h44>
Client-side communication costs are significantly reduced as the data to be
transmitted is restricted to initial layers of the split learning network (splitNN) prior to the split. The
client-side computation costs of learning the weights of the network are also
significantly reduced for the same reason. In terms of model performance, the
accuracies of Split NN remained competitive to other distributed deep learning methods like federated learning and large
batch synchronous SGD with a drastically smaller client side computational
burden when training on a larger number of clients as shown below in terms of teraflops of computation and gigabytes of communication when split learning is used to train Resnet and VGG architectures over 100 and 500 clients with CIFAR 10 and CIFAR 100 datasets. 

<p align="center"><img src="https://splitlearning.github.io/splitTable.png" height="320" width="600"></p>
<p align="center"><img src="https://splitlearning.github.io/splitPlot.png" height="350" width="700"></p>
<br />
<h4> Versatile plug-and-play configurations of split learning </h4>
Versatile configurations of split learning configurations cater to various practical settings of **i) multiple entities holding different modalities of patient data, ii) centralized and local health entities collaborating on
multiple tasks, iii) learning without sharing labels, iv) multi-task split learning, v) multi-hop split learning** and other hybrid possibilities to name a few as shown below and further detailed in our paper here [(PDF)](https://arxiv.org/pdf/1812.00564.pdf "Pdf")
<p align="center"><img src="https://splitlearning.github.io/splitConfig.png" height="350"></p>
<br /><br />


<h3 id="news"> News stories </h3>
**MIT Technology Review:** A new AI method can train on medical records without revealing patient data https://www.technologyreview.com/the-download/612567/a-new-ai-method-can-train-on-medical-records-without-revealing-patient-data/

**MIT Technology Review:** A little-known AI method can train on your health data without threatening your privacy https://www.technologyreview.com/s/613098/a-little-known-ai-method-can-train-on-your-health-data-without-threatening-your-privacy/

**MIT Technology Review:** The Algorithm Newsletter: The privacy-preserving AI technique that will transform healthcare
