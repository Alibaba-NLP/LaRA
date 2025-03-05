<paper 0>
# Evidence of free-bound transitions in warm dense matter and their impact on equation-of-state measurements 

Maximilian P. Böhme ${ }^{1,2,3^{*}}$, Luke B. Fletcher ${ }^{4}$, Tilo Döppner ${ }^{5}$, Dominik<br>Kraus $^{6,2}$, Andrew D. Baczewski ${ }^{7}$, Thomas R. Preston ${ }^{8}$, Michael J.<br>MacDonald ${ }^{5}$, Frank R. Graziani ${ }^{5}$, Zhandos A. Moldabekov ${ }^{1,2}$, Jan Vorberger ${ }^{2}$<br>and Tobias Dornheim ${ }^{1,2^{*}}$<br>${ }^{1}$ Center for Advanced Systems Understanding (CASUS), Görlitz, D-02826, Germany.<br>${ }^{2}$ Helmholtz-Zentrum Dresden-Rossendorf (HZDR), Dresden, D-01328, Germany.<br>${ }^{3}$ Technische Universität Dresden, Dresden, D-01062, Germany.<br>${ }^{4}$ SLAC National Accelerator Laboratory, Menlo Park, California 94309, USA.<br>${ }^{5}$ Lawrence Livermore National Laboratory, Livermore, California 94550, USA.<br>${ }^{6}$ Institut für Physik, Universität Rostock, Rostock, D-18057, Germany.<br>${ }^{7}$ Sandia National Laboratories, Albuquerque, NM 87185, USA.<br>${ }^{8}$ European XFEL, Schenefeld, D-22869, Germany.

*Corresponding authors. E-mail: m.boehme@hzdr.de; t.dornheim@hzdr.de;


#### Abstract

Warm dense matter (WDM) is now routinely created and probed in laboratories around the world, providing unprecedented insights into conditions achieved in stellar atmospheres, planetary interiors, and inertial confinement fusion experiments. However, the interpretation of these experiments is often filtered through models with systematic errors that are difficult to quantify. Due to the simultaneous presence of quantum degeneracy and thermal excitation, processes in which free electrons are de-excited into thermally unoccupied bound states transferring momentum and energy to a scattered x-ray photon become viable. Here we show that such free-bound transitions are a particular feature of WDM and vanish in the limits of cold and hot temperatures. The inclusion of these processes into the analysis of recent X-ray Thomson Scattering experiments on WDM at the National Ignition Facility and the Linac Coherent Light Source significantly improves model fits, indicating that free-bound transitions have been observed without previously being identified. This interpretation is corroborated by agreement with a recently developed model-free thermometry technique and presents an important step for precisely characterizing and understanding the complex WDM state of matter.


Keywords: warm dense matter, laboratory astrophysics, X-ray Thomson scattering, equation-of-state

The study of matter at extreme temperatures $\left(T \sim 10^{3}-10^{8} \mathrm{~K}\right)$ and pressures $(P \sim 1-$ $10^{4}$ Mbar) constitutes a highly active frontier at the interface of a variety of research fields including plasma physics, electronic structure, material science, and scientific computing [1-3]. Such warm dense matter (WDM) occurs in astrophysical
objects [4] such as giant planet interiors [5-7] and brown dwarfs $[8,9]$. For terrestrial applications, WDM is of prime relevance for materials synthesis and discovery, with the recent observation of diamond formation at high pressures [10, 11] being a case in point. Additionally, the WDM regime must be traversed on the way to ignition [12] in inertial confinement fusion (ICF) [13], where recent breakthroughs [14] promise a potential abundance of clean energy in the future.

As a direct consequence of this remarkable interest, WDM is nowadays routinely created in large research centers around the globe, including the European XFEL in Germany [15], SACLA in Japan [16], as well as LCLS [17], the OMEGA laser [18], the Z Pulsed Power Facility [19], and the National Ignition Facility (NIF) [20, 21] in the USA. Yet, the rigorous interpretation of WDM experiments constitutes a formidable challenge. Specifically, the extreme conditions often prevent the direct measurement even of basic parameters such as temperature and density, which have to be inferred from other observations [2]. In this situation, x-ray Thomson scattering (XRTS) [22] has emerged as a powerful and capable tool that provides unprecedented insights into the behaviour of WDM [23-28]. The measured XRTS intensity for a probe energy of $\omega_{0}$ is given by the convolution of the combined source and instrument function $R(\omega)$ with the electronic dynamic structure factor $S_{e e}(\mathbf{q}, \omega), I(\mathbf{q}, \omega)=R(\omega) \circledast S_{e e}\left(\mathbf{q}, \omega_{0}-\omega\right)$, where the latter, in principle, contains the desired physical information about the probed system.

Since the deconvolution of $I(\mathbf{q}, \omega)$ to recover the dynamic structure factor is generally rendered unstable by noise in the experimental data, the standard approach for the interpretation of an XRTS measurement is to 1) construct a model for $S_{e e}(\mathbf{q}, \omega)$ where unknown variables such as the temperature are being treated as fit parameters, 2) convolve the model with $R(\omega)$, and 3 ) determine the model parameters by finding the values such that the convolved model best fits the measured intensity signal. On the one hand, this approach is, in principle, capable of giving access to a variety of properties, such as the equation of state (EOS) [29-31], and the thus inferred information constitutes constraints for EOS tables that impact a gamut of applications relevant to
ICF [12, 32, 33], materials science, and astrophysical models [2]. On the other hand, it is also clear that the inferred parameters can strongly depend on the employed model, which are typically based on a number of assumptions, such as the decomposition of the electronic orbitals into bound and free populations within the widely used Chihara ansatz [23, 34] (see Fig. 1 and its discussion below). While the Chihara ansatz is exact in principle, its practical implementation in computationally efficient models constitutes a source of systematic error. First-principles models for the dynamic structure factor overcome this [35], but the comparably large computational cost might render them impractical.

In fact, the challenges of modelling XRTS signals directly reflect the notorious difficulty of finding a rigorous theoretical description of WDM. Indeed, a key feature of WDM is the intriguingly intricate interplay of a variety of physical effects, including quantum degeneracy (e.g. Pauli blocking), Coulomb coupling, and strong thermal excitations [1, 36, 37]. For temperatures comparable to the energies of bound states, the thermal excitation of electrons out of these states into free states enables x-ray scattering processes in which a free electron is de-excited back into an empty bound state, transferring energy and momentum to a scattered photon.

In the present work, we demonstrate that accounting for these scattering processes in interpreting XRTS experiments [26,38] improves fits based on the Chihara decomposition and brings inferred temperatures into better agreement with model-free temperature estimates [27, 39]. The reinterpretation of these experiments has thus observed free-bound (FB) transitions in hard xray scattering for the first time. As we explain below, such FB transitions are a distinct feature of WDM and vanish in the limits of cold and hot temperatures. Moreover, we show that the incorporation of FB transitions into the nearly universally used Chihara based interpretation of XRTS experiments has a direct and substantial impact on the inferred parameters, which is of immediate consequence for EOS measurements and the wide range of properties inferred from XRTS experiments.

![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-03.jpg?height=412&width=1506&top_left_y=179&top_left_x=192)

Fig. 1 Illustration of the Chihara decomposition of the XRTS signal into different contributions involving effectively bound and free electrons. a) An incident photon (blue) is scattered on a free electron, leading to an energy loss (red-shifted) or energy gain (blue-shift) of the photon; b) scattering on a bound electron changing its state, also leading to either energy loss or gain; c) the top half depicts the scattering on a bound electron that gets lifted to the continuum due to the energy loss of the photon, i.e., a bound-free transition. The bottom half shows the hitherto unaccounted reverse process, i.e., the scattering on a free electron leading to an energy gain for the photon and transferring the electron to a bound state. Such free-bound (FB) transitions are a particular feature of the complex physics emerging at WDM conditions.

Idea. The dynamic structure factor can be conveniently expressed in its exact spectral representation as [40]

$$
\begin{equation*}
S_{e e}(\mathbf{q}, \omega)=\sum_{m, l} P_{m}\left\|n_{m l}(\mathbf{q})\right\|^{2} \delta\left(\omega-\omega_{l m}\right) \tag{1}
\end{equation*}
$$

i.e., as a sum over all possible transitions between the eigenstates $l$ and $m$ of the full electronic Hamiltonian, with $n_{m l}(\mathbf{q})$ being the transition element induced by a density fluctuation of wave vector $\mathbf{q}, \omega_{l m}=\left(E_{l}-E_{m}\right) / \hbar$ the energy difference, and $P_{m}$ the occupation probability of the initial state $m$. Evidently, $S_{e e}(\mathbf{q}, \omega)$ describes transitions where the scattered photon has lost energy to (gained energy from) the electronic system for $\omega>0(\omega<0)$. It is easy to see that the ratio of energy gain to energy loss is given by the simple detailed balance relation $S_{e e}(\mathbf{q},-\omega) / S_{e e}(\mathbf{q}, \omega)=$ $e^{-\hbar \omega / k_{\mathrm{B}} T}$ in thermodynamic equilibrium [40, 41].

The basic idea behind the Chihara ansatz is the decomposition into scattering events involving effectively bound and free electrons. The total electronic dynamic structure factor is then given by $S_{\mathrm{ee}}(\mathbf{q}, \omega)=S_{\mathrm{FF}}(\mathbf{q}, \omega)+S_{\mathrm{BB}}(\mathbf{q}, \omega)+S_{\mathrm{BF}}(\mathbf{q}, \omega)$, and the individual components are explained in Fig. 1. Specifically, the first two terms describe transitions between two states where the scattered electron remains either free (free-free transitions) or transits between two bound states (boundbound transitions). We note that both types of transition can result either in an energy gain or an energy loss of the photon. Finally, $S_{\mathrm{BF}}(\mathbf{q}, \omega)$ describes transitions of bound electrons into the continuum (bound-free transitions), cf. Fig. 1c). Yet Eq. (1) directly implies that the reverse process is also possible: a FB transition, where the scattering of a photon on an initially free electron causes it to transition into a bound state, resulting in a contribution to $S_{e e}(\mathbf{q}, \omega)$ for $\omega<0$. This contribution has been largely ignored in the XRTS-related WDM literature [22, 23, 38], even though it is directly analogous to the scattering processes that give rise to the blue-shifted plasmon peaks essential to XRTS thermometry. We will demonstrate that its impact on the analysis of XRTS measurements can be substantial.

While there are many contexts in physics and chemistry in which electronic transitions between bound and free states contribute to some observable process (e.g., radiative cooling/recombination in plasmas [42], solids [43], or ultracold gases [44]), in the context of XRTS these transitions are signatures of the WDM regime. At low temperatures ( $k_{\mathrm{B}} T \ll E_{\mathrm{F}}$ with $E_{\mathrm{F}}$ being the Fermi energy), the probability to find an electron in any excited free state goes to zero. This leads to an exponential damping of the FB contribution by the detailedbalance factor, $S_{\mathrm{FB}}(\mathbf{q}, \omega)=e^{-\hbar \omega / k_{\mathrm{B}} T} S_{\mathrm{BF}}(\mathbf{q}, \omega)$. Conversely, a given sample can be expected to be fully ionized in the hot-dense matter regime, such that bound-free transitions are not possible: i.e. $S_{\mathrm{BF}}(\mathbf{q}, \omega)=S_{\mathrm{FB}}(\mathbf{q}, \omega)=0$. In other words, the appearance of FB transitions in x-ray scattering is directly interwoven with partial ionization, which is a key feature of WDM systems [1].

![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-04.jpg?height=1197&width=728&top_left_y=199&top_left_x=156)

Fig. 2 XRTS scattering intensity (solid green) of a Be ICF experiment at NIF (N170214) by Döppner et al. [26] with a scattering angle of $\theta=120^{\circ}\left(q=7.89 \AA^{-1}\right)$. The solid red line shows the best fit, and the dotted black, dashed blue, and dash-dotted grey lines show the corresponding components from the Chihara decomposition illustrated in Fig. 1. Top: Chihara fit without taking into account the physically mandated free-bound transitions, with parameters taken from the original Ref. [26]. Bottom: Improved fit from the present work, with the FB contribution being indicated by the shaded blue area.

Results. To demonstrate the practical importance and observation of FB transitions, we first consider a recent experiment at the NIF [26] where beryllium capsules were compressed using 184 of the 192 laser beams. An additional laser-driven xray backlighter source [45] (with zinc He- $\alpha$ lines at $8950 \mathrm{eV}$ and $8999 \mathrm{eV}$ ) was used as a probe for the XRTS measurement, and the results are shown in Fig. 2. The top panel shows the original analysis by Döppner et al. [26] where the

![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-04.jpg?height=632&width=748&top_left_y=184&top_left_x=976)

Fig. 3 Convergence of the extracted temperature for the spectrum shown in Fig. 2 using the ITCF thermometry technique $[27,39]$ with the integration range $x$, see Eq. (3) in the Methods section. Blue: ITCF estimate taking into account the instrument function $R(\omega)$; green: raw ITCF estimate neglecting $R(\omega)$; red: best fit properly taking into account free-bound transitions; orange: best fit from the original Ref. [26] without including the free-bound term.

solid green line depicts the measured XRTS signal, and the solid red line corresponds to the best fit. In addition, the dotted black, dash-dotted grey, and dashed blue lines show the respective components from bound-bound, free-free, and bound-free transitions. Here, the bound-free contribution violates detailed-balance and does not extend to the up-shifted part of the spectrum.

In the bottom panel, we redo this analysis and take into account the FB contribution. We find that the free-bound component (shaded blue) to the total XRTS intensity has a substantial and significant weight, which means that the present analysis constitutes a rigorous observation of this effect in WDM. Furthermore, the inclusion of FB transitions fixes the previously violated, but physically mandated detailed balance condition. This is particularly relevant for the description of the up-shifted part of the spectrum. In the original analysis [26], the agreement between fit and measurement for $\omega>\omega_{0}$ was fully determined by the free-free part of the full spectrum, which is unphysical and necessarily leads to inconsistencies with the corresponding down-shifted side. The best fit shown in the top panel of Fig. 2 thus constitutes a compromise with a nominal temperature of $T=160 \mathrm{eV}$. Incorporating the FB contribution removes this inconsistency, thereby lowering the
extracted temperature to $T=149 \mathrm{eV}$ when FB transitions are included.

While the rigorous benchmarking of such a Chihara-based analysis had been precluded by the lack of exact simulation tools in the past, this bottleneck has recently been partially lifted in Refs. [27, 39] where a highly accurate and modelfree method to extract the temperature from XRTS measurements of WDM has been introduced. This methodology is based on Feynman's acclaimed imaginary-time path-integral formulation of statistical mechanics, and a brief introduction to the underlying idea of imaginary-time correlation function (ITCF) thermometry is presented in the Methods section. The results for the temperature analysis are shown in Fig. 3, where the $y$-axis corresponds to the inverse temperature $\beta=1 / k_{\mathrm{B}} T$ and the $x$-axis to the symmetrically truncated integration range, which is a convergence parameter for the ITCF approach. Moreover, the blue (green) line has been obtained taking into account (not taking into account) the source and instrument function $R(\omega)$, and we find a clear onset of convergence for $x \gtrsim 350 \mathrm{eV}$.

The temperature extracted from the best fit by Döppner et al. [26] (horizontal orange line) is significantly larger than the ITCF estimate $T=149 \pm 10 \mathrm{eV}$ (blue line/shaded region). The physically consistent fit including FB transitions (horizontal red line), on the other hand, is in excellent agreement with the model-free ITCF result. A corresponding analysis of a second XRTS spectrum for Be that has been obtained during the same NIF shot, but at an earlier time (and, thus, an overall lower temperature) is presented in the Methods section.

As a second example, we consider an XRTS measurement of isochorically heated graphite that was performed by Kraus et al. [38] at LCLS. The corresponding XRTS signal is shown as the solid green curve in Fig. 4, and the top panel corresponds to the original analysis from Ref. [38]. Evidently, the bound-free feature due to the excitation of weakly bound L-shell electrons constitutes the dominant contribution to the inelastic feature around $\omega=5800 \mathrm{eV}$; it is, however, absent from the up-shifted part. As a consequence, the thus constructed model for the full XRTS intensity strongly violates the detailed balance between energy loss and energy gain. Moreover, the upshifted feature is fitted entirely by the free-free
![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-05.jpg?height=1198&width=748&top_left_y=201&top_left_x=996)

Fig. 4 XRTS scattering intensity (solid green) of isochorically heated graphite at a scattering angle of $\theta=160^{\circ}$ ( $q=5.9 \AA^{-1}$ ) obtained by Kraus et al. [38] at LCLS. The solid red line shows the best fit, and the solid black, dashed blue, and dash-dotted grey lines show the corresponding components from the Chihara decomposition illustrated in Fig. 1. Top: Chihara fit without taking into account the physically mandated free-bound transitions, with parameters taken from the original Ref. [38]. Bottom: Improved fit from the present work, with the FB contribution being indicated by the shaded blue area.

contribution, leading to similar inconsistencies as reported for the NIF Be shot investigated in Fig. 2 above. In the bottom panel of Fig. 4, we show our present analysis based on the improved, physically consistent Chihara model that includes FB transitions; the latter are highlighted as the shaded blue area in the up-shifted part of the spectrum. Our new fit confers a substantial improvement compared to the original fit over the entire frequency range. The temperature that we have extracted from the best fit is given by $T=16.6 \mathrm{eV}$, which is
in excellent agreement with the ITCF-based value of $T=18 \pm 2 \mathrm{eV}$ given in Ref. [39] that has been obtained without any model assumptions; see the Methods section for additional details. In contrast, the best fit by Kraus et al. [38] gives $T=21.7 \mathrm{eV}$, and thus significantly overestimates the temperature. This again highlights the importance of FB transitions for the determination of the equationof-state of materials at WDM conditions based on XRTS measurements.

Discussion. In this work, we have unambiguously demonstrated the importance of including free-bound transitions in models of XRTS signals, both to restore physically mandated detailed balance and reduce sources of systematic error. The prominence of signatures of these processes is a direct consequence of the complex interplay of various physical effects emerging in WDM; it vanishes both in the limits of low temperatures and in the hot dense matter regime. This is illustrated in the top panel of Fig. 5, where we show the ionization degree of Be as a function of temperature and mass density. In particular, FB transitions are significant in the bright area around the center, i.e., above a threshold of $T=50 \mathrm{eV}$ due to the detailed balance factor, and below full ionization at around $T=150-400 \mathrm{eV}$ (bottom right), depending on $\rho$.

To demonstrate the importance of this effect, we have re-analyzed two representative XRTS experiments with WDM: the recent investigation of strongly compressed Be at the NIF by Döppner et al. [26], and an investigation of isochorically heated graphite at LCLS by Kraus et al. [38]. In both cases, we have found that including freebound transitions leads to a reduction of the inferred temperature, and an improved agreement between theory and measurement. These findings have been further substantiated by the excellent agreement of the thus extracted temperature with the independent and model-free ITCF thermometry technique developed in Refs. [27, 39]. To put our results into the proper context, we show an overview of a number of XRTS experiments with different elements in the WDM regime in Fig. 5b). The different symbols distinguish different materials, and the dotted vertical green (red) lines show the temperature above which FB transitions become important for the respective $\mathrm{L}$ edge (K edge). The experiments with Be (cf. Fig. 2) and carbon (cf. Fig. 4), as reanalyzed here, are
![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-06.jpg?height=1256&width=724&top_left_y=179&top_left_x=998)

Fig. 5 Top: Ionization degree of beryllium computed by FLYCHK [46]. FB transitions are important for partial ionization, i.e., in the bright area around the center; they are suppressed by the detailed balance relation for $T \lesssim 50 \mathrm{eV}$ (left), and by full ionization for $T \gtrsim 150-400 \mathrm{eV}$ depending on $\rho$ (bottom right). Bottom: overview of selected XRTS experiments with different elements (see point style) in the WDM regime, shown in the temperature-number density plane. The vertical dotted green (red) lines indicate the minimum temperature above which FB transitions become significant for the respective $\mathrm{L}$ edge (K edge). The shaded green, yellow, and red areas correspond to experimental capabilities at the European XFEL [15], the brandnew Colliding Planar Shock platform (CPS) at NIF [21], and the Gigabar-XRTS platform at NIF [26]. The red line shows a simulated NIF ICF implosion path.

indicated by the blue diamond and square, respectively. Furthermore, the shaded areas indicate the experimental capabilities at two key facilities. The shaded green bubble corresponds to the European XFEL [15], which combines excellent beam properties with an unprecedented repetition rate for pump-probe experiments. The latter
allows for highly accurate XRTS measurements over a dynamic range of at least three orders of magnitude in the intensity [47], which makes it possible to resolve FB transitions even for $T \sim$ $10 \mathrm{eV}$. The red circle outlines parameters realized at the NIF Gbar-XRTS platform [26], where FB transitions indeed constitute a predominant feature; cf. our re-analysis of the Be experiment by Döppner et al. [26] in Fig. 2 above. Of particular importance is the olive square corresponding to the new Colliding Planar Shock (CPS) platform at NIF [21]. It has been designed to realize very uniform conditions that are ideally suited for precision EOS measurements. Yet, the accurate determination of the latter requires FB transitions to be taken into account for the interpretation of XRTS experiments, as it can be clearly seen from Fig. 5 .

The work presented here will have a direct and profound impact on a number of research fields related to the study of matter under extreme conditions. XRTS constitutes a widely used method of diagnostics for WDM and beyond. This makes it particularly important for benchmarking EOS models [31], which constitute a key input for understanding astrophysical phenomena [2], fusion applications $[12,32]$, and a plethora of other calculations [48]. The particular relevance of FB transitions to fusion applications is further substantiated by a simulated ICF implosion path that has been included as the solid red line in the bottom panel of Fig. 5.

Furthermore, incorporating free-bound transitions into the Chihara model restores the exact detailed balance relation. This opens up the way for the systematic improvement of the underlying theoretical description of the individual components (cf. Fig. 1), which, otherwise, might have been biased by the artificial distortion between the positive and negative frequency range; this would likely have prevented agreement between theory and experiment even for an exact description of $S_{\mathrm{FF}}(\mathbf{q}, \omega), S_{\mathrm{BB}}(\mathbf{q}, \omega)$, and $S_{\mathrm{BF}}(\mathbf{q}, \omega)$ in many cases.

A particularly enticing route for future research is given by the application of the improved, physically consistent Chihara model to emerging exact path integral Monte Carlo simulation results for warm dense hydrogen [49] and other light elements. These efforts will allow us to rigorously benchmark existing models for the individual contributions to $S_{e e}(\mathbf{q}, \omega)$, and to guide the development of improved theories. Moreover, such a comparison will allow us to assess the conceptual validity of the decomposition into bound and free electrons on which the Chihara model is based, which will be of fundamental importance to our understanding of electron-ion systems in different contexts.

## Methods

Model-free imaginary-time correlation function temperature diagnostics. In the main text, we have demonstrated the improved accuracy of the Chihara fit including the freebound transitions by comparing the extracted temperature with the highly accurate and modelfree imaginary-time correlation function (ITCF) thermometry technique $[27,39]$. The basic idea is to consider the two-sided Laplace transform of $S(\mathbf{q}, \omega)$, which gives the imaginary-time version of the usual intermediate scattering function $F_{e e}(\mathbf{q}, t)[22]$,

$$
\begin{equation*}
F_{e e}(\mathbf{q}, \tau)=\int_{-\infty}^{\infty} \mathrm{d} \omega S_{e e}(\mathbf{q}, \omega) e^{-\tau \hbar \omega} \tag{2}
\end{equation*}
$$

Specifically, the time argument has been replaced by $t=-i \hbar \tau$, with $\tau \in[0, \beta]$ and $\beta=1 / k_{\mathrm{B}} T$. From a mathematical perspective, both the $\tau$ and the $\omega$-representation are formally equivalent. It has recently been pointed out that working in the imaginary-time domain has a number of key advantages [27, 37, 39]. First, the deconvolution with respect to the source-and-instrument function $R(\omega)$ becomes straightforward even in the presence of substantial noise in the experimental data. Second, the detailed balance relation connecting $S_{e e}(\mathbf{q}, \omega)$ with $S_{e e}(\mathbf{q},-\omega)$ leads to the symmetry relation $F_{e e}(\mathbf{q}, \tau)=F_{e e}(\mathbf{q}, \beta-\tau)$, which always manifests as a minimum of $F_{e e}(\mathbf{q}, \tau)$ around $\tau=\beta / 2=1 / 2 T$. In other words, locating the minimum in the ITCF Eq. (2) gives one direct access to the temperature of an arbitrarily complex system without any simulations or approximations.

An additional difficulty is given by the necessarily finite spectral range of any experimental data set, whereas the evaluation of Eq. (2), in principle, would require an integration from negative to positive infinity. In practice, we compute the

![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-08.jpg?height=1196&width=725&top_left_y=202&top_left_x=155)

Fig. 6 XRTS scattering intensity (solid green) of the same Be ICF experiment at NIF (N170214) by Döppner et al. [26] shown in Fig. 2 in the main text, but at an earlier probe time where the capsule is less compressed and less heated. The solid red line shows the best fit, and the dotted black, dashed blue, and dash-dotted grey lines show the corresponding components from the Chihara decomposition illustrated in Fig. 1. Top: Chihara fit without taking into account the physically mandated free-bound transitions, with parameters taken from the original Ref. [26]. Bottom: Improved fit from the present work, with the FB contribution being indicated by the shaded blue area.

symmetrically truncated ITCF

$$
\begin{equation*}
F_{x}(\mathbf{q}, \tau)=\int_{-x}^{x} \mathrm{~d} \omega S_{e e}(\mathbf{q}, \omega) e^{-\tau \hbar \omega} \tag{3}
\end{equation*}
$$

and the convergence of the thus extracted temperature with the integration range $x$ is demonstrated in Fig. 3 in the main text.

![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-08.jpg?height=615&width=744&top_left_y=200&top_left_x=978)

Fig. 7 Convergence of the extracted temperature for the spectrum shown in Fig. 6 using the ITCF thermometry technique $[27,39]$ with the integration range $x$, see Eq. (3) in the Methods section. Blue: ITCF estimate taking into account the instrument function $R(\omega)$; green: raw ITCF estimate neglecting $R(\omega)$; red: best fit properly taking into account free-bound transitions; orange: best fit from the original Ref. [26] without including the free-bound term.

## Analysis of complementary NIF spectrum

 at lower temperature. To further demonstrate the general nature of FB transitions for the interpretation of XRTS experiments probing WDM states, we re-analyze a second measurement of compressed Be in Fig. 6. It has been obtained during the same experiment (N170214) as the spectrum shown in Fig. 2 in the main text, but at an earlier time, leading to a comparably reduced degree of compression and a lower temperature. The top and bottom panels again show the Chihara-based analysis without and with including the FB contribution, and we find the same trends as in the previous examples.The corresponding ITCF analysis of the extracted temperature is shown in Fig. 7. The best fit from the original Ref. [26] gives a temperature of $T=110 \mathrm{eV}$ (orange), whereas our improved model that includes FB transitions lowers that value to $T=97 \mathrm{eV}$ (red). The latter is in significantly better agreement with the properly deconvolved ITCF result of $T=99 \pm 7 \mathrm{eV}$ (blue), as it is expected.

Impact of the instrument function model. An important aspect of the rigorous interpretation of XRTS experiments is the characterization

![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-09.jpg?height=610&width=710&top_left_y=205&top_left_x=172)

Fig. 8 Re-analyzing the XRTS experiment shown in Fig. 6 with a modified, empirical model for the combined source and instrument function.

of the combined source and instrument function [45]. This holds both for the model-free ITCF method [27, 39] discussed above, and for the Chihara-based forward modelling approach that constitutes the focus of the present work. In Fig. 8, we have repeated the analysis of the colder Be NIF spectrum shown in the bottom panel of Fig. 6, but using an empirical model for the instrument function that has been obtained by fitting FLYCHK emission lines to the elastic feature directly extracted from the measured XRTS signal. The main differences to the source function model used in the original Ref. [26] are the absence of the left shoulder, and a somewhat less pronounced double peak structure.

Evidently, this empirical, data-driven source function leads to a substantially improved agreement between the Chihara model and the experimental data. The corresponding temperature analysis is shown in Fig. 9, and we find the same trends as in the previous section.

This leads us to the following conclusions: i) residual deviations between the improved Chihara model that includes FB transitions and experimental measurements can be a consequence either of systematic inaccuracies in the individual components of the former, or of the employed model for the source and instrument function; ii) the quantification of uncertainties in the source function, and the systematic study of their impact on both forward modelling and ITCF based analysis frameworks constitutes an important task for

![](https://cdn.mathpix.com/cropped/2024_06_04_ee6dd09a0e52fab62e06g-09.jpg?height=615&width=745&top_left_y=200&top_left_x=995)

Fig. 9 Convergence of the extracted temperature for the spectrum shown in Fig. 8 using the ITCF thermometry technique $[27,39]$ with the integration range $x$, see Eq. (3) in the Methods section. Blue: ITCF estimate taking into account the instrument function $R(\omega)$; green: raw ITCF estimate neglecting $R(\omega)$; red: best fit properly taking into account free-bound transitions; orange: best fit from the original Ref. [26] without including the free-bound term.

future work; iii) the observation of the physically mandated FB transitions reported in the present work is very robust and does not depend on any particular model.

## Data Availability

The spectra generated from our improved Chihara model that includes free-bound transitions will be made openly available in the Rossendorf data repository (RODARE).

## References

[1] Graziani F, Desjarlais MP, Redmer R, Trickey SB, editors. Frontiers and Challenges in Warm Dense Matter. International Publishing: Springer; 2014.

[2] Drake RP. High-Energy-Density Physics: Foundation of Inertial Fusion and Experimental Astrophysics. Graduate Texts in Physics. Springer International Publishing; 2018 .

[3] Hatfield PW, Gaffney JA, Anderson GJ, Ali S, Antonelli L, Başeğmez du Pree $S$, et al. The data-driven future of
high-energy-density physics. Nature. 2021 May;593(7859):351-361. https://doi.org/10. 1038/s41586-021-03382-w.

[4] Bailey JE, Nagayama T, Loisel GP, Rochau GA, Blancard C, Colgan J, et al. A higherthan-predicted measurement of iron opacity at solar interior temperatures. Nature. 2015 Jan;517(7532):56-59. https://doi.org/ 10.1038/nature14048.

[5] Liu SF, Hori Y, Müller S, Zheng X, Helled R, Lin D, et al. The formation of Jupiter's diluted core by a giant impact. Nature. 2019 Aug;572(7769):355-357. https://doi.org/10. 1038/s41586-019-1470-2.

[6] Brygoo S, Loubeyre P, Millot M, Rygg JR, Celliers PM, Eggert JH, et al. Evidence of hydrogen-helium immiscibility at Jupiter-interior conditions. Nature. 2021 May;593(7860):517-521. https://doi.org/10. 1038/s41586-021-03516-0.

[7] Kraus RG, Hemley RJ, Ali SJ, Belof JL, Benedict LX, Bernier J, et al. Measuring the melting curve of iron at super-Earth core conditions. Science. 2022;375(6577):202-205. https://doi.org/10.1126/science.abm1472.

[8] Kritcher AL, Swift DC, Döppner T, Bachmann B, Benedict LX, Collins GW, et al. A measurement of the equation of state of carbon envelopes of white dwarfs. Nature. 2020 Aug;584(7819):51-54. https://doi.org/ $10.1038 / \mathrm{s} 41586-020-2535-\mathrm{y}$.

[9] Becker A, Lorenzen W, Fortney JJ, Nettelmann N, Schöttler M, Redmer R. Ab initio equations of state for hydrogen (HREOS.3) and helium (He-REOS.3) and their implications for the interior of brown dwarfs. Astrophys J Suppl Ser. 2014;215:21. https: //doi.org/10.1088/0067-0049/215/2/21.

[10] Kraus D, Ravasio A, Gauthier M, Gericke DO, Vorberger J, Frydrych S, et al. Nanosecond formation of diamond and lonsdaleite by shock compression of graphite. Nature Communications. 2016 Mar;7(1):10970. https:// doi.org /10.1038/ncomms10970.
[11] Kraus D, Vorberger J, Pak A, Hartley NJ, Fletcher LB, Frydrych S, et al. Formation of diamonds in laser-compressed hydrocarbons at planetary interior conditions. Nature Astronomy. 2017 Sep;1(9):606-611. https: //doi.org/10.1038/s41550-017-0219-9.

[12] Hu SX, Militzer B, Goncharov VN, Skupsky S. First-principles equation-of-state table of deuterium for inertial confinement fusion applications. Phys Rev B. 2011 Dec;84:224109. https://doi.org/10. 1103/PhysRevB.84.224109.

[13] Betti R, Hurricane OA. Inertialconfinement fusion with lasers. Nature Physics. $2016 \quad$ May;12(5):435-448. https://doi.org/10.1038/nphys3736.

[14] Zylstra AB, Hurricane OA, Callahan DA, Kritcher AL, Ralph JE, Robey HF, et al. Burning plasma achieved in inertial fusion. Nature. 2022 Jan;601(7894):542-548. https: //doi.org/10.1038/s41586-021-04281-w.

[15] Tschentscher T, Bressler C, Grünert J, Madsen A, Mancuso AP, Meyer M, et al. Photon Beam Transport and Scientific Instruments at the European XFEL. Applied Sciences. 2017;7(6). https://doi.org/10.3390/ app7060592.

[16] Pile D. First light from SACLA. Nature Photonics. 2011 Aug;5(8):456-457. https://doi. org/10.1038/nphoton.2011.178.

[17] Bostedt C, Boutet S, Fritz DM, Huang Z, Lee HJ, Lemke HT, et al. Linac Coherent Light Source: The first five years. Rev Mod Phys. 2016 Mar;88:015007. https://doi.org/ 10.1103/RevModPhys.88.015007.

[18] Soures JM, McCrory RL, Verdon CP, Babushkin A, Bahr RE, Boehly TR, et al. Direct-drive laser-fusion experiments with the OMEGA, 60-beam, $>40 \mathrm{~kJ}$, ultraviolet laser system. Physics of Plasmas. 1996 05;3(5):2108-2112. https://doi.org/10.1063/ 1.871662 .

[19] Sinars D, Sweeney M, Alexander C, Ampleford D, Ao T, Apruzese J, et al. Review
of pulsed power-driven high energy density physics research on Z at Sandia. Physics of Plasmas. 2020;27(7). https://doi.org/10. $1063 / 5.0007476$.

[20] Moses EI, Boyd RN, Remington BA, Keane CJ, Al-Ayat R. The National Ignition Facility: Ushering in a new age for high energy density science. Physics of Plasmas. 2009;16(4):041006. https://doi.org/10.1063/ 1.3116505 .

[21] MacDonald MJ, Di Stefano CA, Döppner T, Fletcher LB, Flippo KA, Kalantar D, et al. The colliding planar shocks platform to study warm dense matter at the National Ignition Facility. Physics of Plasmas. 2023 06;30(6). 062701. https://doi.org/10.1063/5.0146624.

[22] Glenzer SH, Redmer R. X-ray Thomson scattering in high energy density plasmas. Rev Mod Phys. 2009;81:1625. https://doi.org/10. 1103/RevModPhys.81.1625.

[23] Gregori G, Glenzer SH, Rozmus W, Lee RW, Landen OL. Theoretical model of x-ray scattering as a dense matter probe. Phys Rev E. 2003 Feb;67:026412. https://doi.org/10. 1103/PhysRevE.67.026412.

[24] Harbour L, Dharma-wardana MWC, Klug DD, Lewis LJ. Pair potentials for warm dense matter and their application to x-ray Thomson scattering in aluminum and beryllium. Phys Rev E. 2016 Nov;94:053211. https: //doi.org/10.1103/PhysRevE.94.053211.

[25] García Saiz E, Gregori G, Gericke DO, Vorberger J, Barbrel B, Clarke RJ, et al. Probing warm dense lithium by inelastic X-ray scattering. Nature Physics. 2008 Dec;4(12):940944. https://doi.org/10.1038/nphys1103.

[26] Döppner T, Bethkenhagen M, Kraus D, Neumayer P, Chapman DA, Bachmann B, et al. Observing the onset of pressure-driven Kshell delocalization. Nature. 2023 May;https: //doi.org/10.1038/s41586-023-05996-8.

[27] Dornheim T, Böhme M, Kraus D, Döppner T, Preston TR, Moldabekov ZA, et al. Accurate temperature diagnostics for matter under extreme conditions. Nature Communications. 2022 Dec;13(1):7911. https://doi.org/ $10.1038 / \mathrm{s} 41467-022-35578-7$.

[28] Dornheim T, Döppner T, Baczewski AD, Tolias P, Böhme MP, Moldabekov ZA, et al. X-ray Thomson scattering absolute intensity from the f-sum rule in the imaginary-time domain. arXiv. 2023;https://arxiv.org/abs/ 2305.15305. [physics.plasm-ph].

[29] Regan SP, Falk K, Gregori G, Radha PB, Hu SX, Boehly TR, et al. Inelastic X-Ray Scattering from Shocked Liquid Deuterium. Phys Rev Lett. 2012 Dec;109:265003. https: //doi.org/10.1103/PhysRevLett.109.265003.

[30] Falk K, Regan SP, Vorberger J, Crowley BJB, Glenzer SH, Hu SX, et al. Comparison between x-ray scattering and velocity-interferometry measurements from shocked liquid deuterium. Phys Rev E. 2013 Apr;87:043112. https://doi.org/10. 1103/PhysRevE.87.043112.

[31] Falk K, Gamboa EJ, Kagan G, Montgomery DS, Srinivasan B, Tzeferacos P, et al. Equation of State Measurements of Warm Dense Carbon Using Laser-Driven Shock and Release Technique. Phys Rev Lett. 2014 Apr;112:155003. https://doi.org/10.1103/ PhysRevLett.112.155003.

[32] Caillabet L, Canaud B, Salin G, Mazevet S, Loubeyre P. Change in Inertial Confinement Fusion Implosions upon Using an Ab Initio Multiphase DT Equation of State. Phys Rev Lett. 2011 Sep;107:115004. https://doi.org/ 10.1103/PhysRevLett.107.115004.

[33] Hurricane OA, Callahan DA, Casey DT, Celliers PM, Cerjan C, Dewald EL, et al. Fuel gain exceeding unity in an inertially confined fusion implosion. Nature. 2014 Feb;506(7488):343-348. https://doi.org/10. 1038/nature13008.

[34] Chihara J. Difference in X-ray scattering between metallic and non-metallic liquids due to conduction electrons. Journal of Physics F: Metal Physics. 1987 feb;17(2):295-304. https: //doi.org/10.1088/0305-4608/17/2/002.

[35] Baczewski AD, Shulenburger L, Desjarlais MP, Hansen SB, Magyar RJ. X-ray Thomson Scattering in Warm Dense Matter without the Chihara Decomposition. Phys Rev Lett. 2016 Mar;116:115004. https://doi.org/10. 1103/PhysRevLett.116.115004.

[36] Bonitz M, Dornheim T, Moldabekov ZA, Zhang S, Hamann P, Kählert H, et al. Ab initio simulation of warm dense matter. Physics of Plasmas. 2020;27(4):042710. https://doi. org/10.1063/1.5143225.

[37] Dornheim T, Moldabekov ZA, Ramakrishna K, Tolias P, Baczewski AD, Kraus D, et al. Electronic density response of warm dense matter. Physics of Plasmas. 2023 03;30(3). 032705. https://doi.org/10.1063/5.0138955.

[38] Kraus D, Bachmann B, Barbrel B, Falcone RW, Fletcher LB, Frydrych S, et al. Characterizing the ionization potential depression in dense carbon plasmas with highprecision spectrally resolved x-ray scattering. Plasma Physics and Controlled Fusion. 2018 nov;61(1):014015. https://doi.org/10.1088/ 1361-6587/aadd6c.

[39] Dornheim T, Böhme MP, Chapman DA, Kraus D, Preston TR, Moldabekov ZA, et al. Imaginary-time correlation function thermometry: A new, high-accuracy and modelfree temperature analysis technique for x-ray Thomson scattering data. Physics of Plasmas. 2023 04;30(4). 042707. https://doi.org/ $10.1063 / 5.0139560$.

[40] Giuliani G, Vignale G. Quantum Theory of the Electron Liquid. Cambridge: Cambridge University Press; 2008.

[41] Döppner T, Landen OL, Lee HJ, Neumayer P, Regan SP, Glenzer SH. Temperature measurement through detailed balance in x-ray Thomson scattering. High Energy Density Physics. 2009;5(3):182-186. https://doi.org/ 10.1016/j.hedp.2009.05.012.

[42] Raymond JC, Cox DP, Smith BW. Radiative cooling of a low-density plasma. The Astrophysical Journal. 1976;204:290-292. https: //doi.org/10.1086/154170.
[43] Van Roosbroeck W, Shockley W. Photonradiative recombination of electrons and holes in germanium. Physical Review. 1954;94(6):1558. https://doi.org/10.1103/ PhysRev.94.1558.

[44] Thorsheim H, Weiner J, Julienne PS. Laser-induced photoassociation of ultracold sodium atoms. Physical review letters. 1987;58(23):2420. https://doi.org/10.1103/ PhysRevLett.58.2420.

[45] MacDonald MJ, Saunders AM, Bachmann B, Bethkenhagen M, Divol L, Doyle MD, et al. Demonstration of a laser-driven, narrow spectral bandwidth x-ray source for collective x-ray scattering experiments. Physics of Plasmas. 2021;28(3):032708. https://doi.org/10. $1063 / 5.0030958$.

[46] Chung HK, Chen MH, Morgan WL, Ralchenko Y, Lee RW. FLYCHK: Generalized population kinetics and spectral model for rapid spectroscopic analysis for all elements. High Energy Density Physics. 2005;1(1):3-12. https://doi.org/10.1016/j.hedp.2005.07.001.

[47] Voigt K, Zhang M, Ramakrishna K, Amouretti A, Appel K, Brambrink E, et al. Demonstration of an x-ray Raman spectroscopy setup to study warm dense carbon at the high energy density instrument of European XFEL. Physics of Plasmas. 2021;28(8):082701. https://doi.org/10.1063/5.0048150.

[48] Militzer B, González-Cataldo F, Zhang S, Driver KP, Soubiran Fmc. First-principles equation of state database for warm dense matter computation. Phys Rev E. 2021 Jan;103:013203. https://doi.org/10.1103/ PhysRevE.103.013203.

[49] Böhme M, Moldabekov ZA, Vorberger J, Dornheim T. Static Electronic Density Response of Warm Dense Hydrogen: Ab Initio Path Integral Monte Carlo Simulations. Phys Rev Lett. 2022 Aug;129:066402. https: //doi.org/10.1103/PhysRevLett.129.066402.
</end of paper 0>


<paper 1>
# Unraveling electronic correlations in warm dense quantum plasmas 

T. Dornheim, ${ }^{1,2 *}$ T. Döppner, ${ }^{3}$ P. Tolias, ${ }^{4}$<br>M. P. Böhme,,$^{1,2,5}$ L.B. Fletcher, ${ }^{6}$ Th. Gawne, ${ }^{1,2}$ F. R. Graziani, ${ }^{3}$<br>D. Kraus, ${ }^{7,2}$ M. J. MacDonald, ${ }^{3}$ Zh. A. Moldabekov, ${ }^{1,2}$<br>S. Schwalbe, ${ }^{1,2}$ D.O. Gericke, ${ }^{8}$ and J. Vorberger ${ }^{2}$<br>${ }^{1}$ Center for Advanced Systems Understanding (CASUS), D-02826 Görlitz, Germany<br>${ }^{2}$ Helmholtz-Zentrum Dresden-Rossendorf (HZDR), D-01328 Dresden, Germany<br>${ }^{3}$ Lawrence Livermore National Laboratory (LLNL), California 94550 Livermore, USA<br>${ }^{4}$ Space and Plasma Physics, Royal Institute of Technology (KTH)<br>Stockholm, SE-100 44, Sweden<br>${ }^{5}$ Technische Universität Dresden, D-01062 Dresden, Germany<br>${ }^{6}$ SLAC National Accelerator Laboratory, CA 94025 Menlo Park, USA<br>${ }^{7}$ Institut für Physik, Universität Rostock, D-18057 Rostock, Germany<br>${ }^{8}$ Centre for Fusion, Space and Astrophysics, University of Warwick, Coventry CV4 7AL, UK

*To whom correspondence should be addressed; E-mail: t.dornheim @hzdr.de

The study of matter at extreme densities and temperatures has emerged as a highly active frontier at the interface of plasma physics, material science and quantum chemistry with direct relevance for planetary modeling and inertial confinement fusion. A particular feature of such warm dense matter is the complex interplay of strong Coulomb interactions, quantum effects, and thermal excitations, rendering its rigorous theoretical description a formidable challenge. Here, we report a breakthrough in path integral Monte Carlo sim-
ulations that allows us to unravel this intricate interplay for light elements without nodal restrictions. This new capability gives us access to electronic correlations previously unattainable. As an example, we apply our method to strongly compressed beryllium to describe x-ray Thomson scattering (XRTS) data obtained at the National Ignition Facility. We find excellent agreement between simulation and experiment. Our analysis shows an unprecedented level of consistency for independent observations without the need for any empirical input parameters.

Matter at extreme densities and temperatures displays a complex quantum behavior. A particularly intriguing situation emerges when the interaction, thermal, and Fermi energies are comparable. Understanding such warm dense matter (WDM) requires a holistic description taking into account partial ionization, partial quantum degeneracy, and moderate coupling. Indeed, even familiar concepts such as well-defined electronic bound states and ionization break down in this regime.

Interestingly, such conditions are widespread throughout the universe, naturally occurring in a host of astrophysical objects such as giant planet interiors (1), brown and white dwarfs (2), and, on Earth, meteor impacts (3). Moreover, WDM plays a key role in cutting-edge technological applications such as the discovery and synthesis of novel materials (4). An extraordinary achievement has recently been accomplished in the field of inertial confinement fusion at the National Ignition Facility (NIF) (5) 6). In these experiments, both the ablator and the fuel traverse the WDM regime, making a rigorous understanding of such states paramount to reach the reported burning plasma and net energy gain (7,8).

The pressing need to understand extreme states has driven a large leap in the experimental capabilities; the considerable number of remarkable successes includes the demonstration of diamond formation under planetary conditions (4,9), opacity measurements under solar condi-
tions (10), probing atomic physics at Gigabar pressures (11), and the determination of energy loss of charged particles $(12,13)$. However, this progress is severely hampered: to diagnose WDM experiments, a thorough understanding of the electronic response is indispensable. Indeed, even the inference of basic parameters such as temperature and density requires rigorous modeling to interpret the probe signal.

Density functional theory combined with classical molecular dynamics for the ions (DFTMD) has emerged as the de-facto work horse for computing WDM properties. While being formally exact (14), the predictive capability of DFT-MD is limited by the unknown exchangecorrelation functional, which has to be approximated in practice, and the application of the Born-Oppenheimer approximation. A potentially superior alternative is given by $a b$ initio path integral Monte Carlo (PIMC) simulations (15), which are in principle capable of providing an exact solution for a given quantum many-body problem without any empirical input. Yet, PIMC simulations of quantum degenerate Fermi systems, such as the electrons in WDM, are afflicted with an exponential computational bottleneck, which is known as the fermion sign problem (16. 17). As a consequence, PIMC application to matter under extreme conditions has either been limited to comparably simple systems such as the uniform electron gas model $(\sqrt{18})$, or based on inherently uncontrolled approximations as in the case of restricted PIMC $(19,20,21)$.

Here, we present a solution to this unsatisfactory situation and demonstrate its capabilities on the example of warm dense beryllium (Be). Since our approach is not based on any nodal restriction, we get access to the full spectral information in the imaginary-time domain (22). As the capstone of our work, we employ our simulations to re-analyze X-ray Thomson scattering (XRTS) data obtained at the NIF for strongly compressed Be in a backscattering geometry (11). In addition, we consider a new data set that has been measured at a smaller scattering angle that is more sensitive to electronic correlations. Our unique access to electron correlation functions allows for novel ways to interpret the XRTS data, resulting in an unprecedented level of con-
sistency. We are convinced that our work will open up a wealth of new avenues for substantial advances of our understanding of warm dense quantum plasmas.

Simulation approach. In principle, the PIMC method allows one to obtain an exact solution to the full quantum-many body problem without any empirical input or approximations. However, the application of PIMC to quantum degenerate electrons is severely hampered by the fermion sign problem (16, 17). To circumvent this obstacle, Militzer, Ceperley and others $19,20,21,23)$ have employed the fixed-node approximation. This restricted PIMC method allows for simulations of large systems without a sign problem, an advantage that comes at the cost of a de-facto uncontrolled approximation (24, 25). Moreover, the implementation of a nodal restriction prevents the usual access of PIMC to the full spectral information in the imaginary-time domain $(26,22)$, preventing a direct comparison with XRTS measurements.

In this work, we employ a fundamentally different strategy by carrying out a controlled extrapolation over a continuous variable $\xi \in[-1,1]$ that is substituted into the canonical partition function (27, 28, 29), see the Supplemental Material (30). This treatment removes the exponential scaling of the computation time with the system size for substantial parts of the WDM regime without the need for any empirical input such as the nodal surface of the density matrix for restricted PIMC or the XC functional for DFT. At the same time, it retains full access to the spectral information about the system encoded in the imaginary-time density-density correlation function (ITCF), thereby allowing for direct comparison between simulations and XRTS measurements. While this approach had been successfully applied to the uniform electron gas model (28 29), we use it here to study the substantially more complex case of electrons and nuclei in WDM for the first time.

In Figs. 1a) and b), we show snapshots of all-electron PIMC simulations of $N_{\mathrm{Be}}=25$ Be atoms (i.e, $N_{\mathrm{e}}=100$ electrons) for the mass density $\rho=7.5 \mathrm{~g} / \mathrm{cc}$ and temperatures of
![](https://cdn.mathpix.com/cropped/2024_06_04_503fd8a2efcadbf2d272g-05.jpg?height=1028&width=1594&top_left_y=662&top_left_x=255)

Figure 1: Ab initio PIMC simulations of compressed Be ( $\rho=7.5 \mathrm{~g} / \mathrm{cc}$ ). a) Snapshot of a PIMC simulation of $N_{\mathrm{Be}}=25$ Be atoms at $T=190 \mathrm{eV}$. b) same as a) but for $T=100 \mathrm{eV}$. The green orbs show the ions and the blue-red paths the quantum degenerate electrons; c),d) electronic density in real space for a fixed ion configuration at the same temperatures. e) Our PIMC simulations give us access to all many-body correlations in the systems, including the spin-resolved electron-electron pair correlation functions $g_{\uparrow \uparrow}(r)$ and $g_{\uparrow \downarrow}(r)$, and the ion-ion pair correlation function $g_{I I}(r)$. f) Electron-ion and ion-ion static structure factors $S_{e I}(q)$ and $S_{I I}(q)$, giving us access to the ratio of elastic and inelastic contributions to the full scattering intensity, see the main text.
$T=190 \mathrm{eV}$ and $T=100 \mathrm{eV}$, respectively. The green orbs depict the nuclei, which behave basically as classical point particles, although this is not built into our simulations. The blue paths represent the quantum degenerate electrons; their extension is proportional to the thermal de Broglie wavelength $\lambda_{T}=\hbar \sqrt{2 \pi / m_{\mathrm{e}} k_{\mathrm{B}} T}$ and serves as a measure for the average extension of a hypothetical single-electron wave function. The interplay of electron delocalization with effects such as Coulomb coupling shapes the physical behavior of the system. In panels c) and d), we illustrate PIMC results for the spatially resolved electron density in the external potential of a fixed ion configuration. We find a substantially increased localization around the nuclei for the lower temperature.

Figs. 1e) and f) show ab initio PIMC results for the full Be system, where both electrons and nuclei are treated dynamically on the same level. Specifically, panel e) shows various pair correlation functions, where the red and green lines correspond to $T=100 \mathrm{eV}$ and $T=190 \mathrm{eV}$, respectively. The ion-ion pair correlation function $g_{I I}(r)$ [squares] is relatively featureless in both cases. The same holds for the spin-diagonal electron-electron pair correlation function $g_{\uparrow \uparrow}(r)=g_{\downarrow \downarrow}(r)$ [crosses], although the exchange-correlation hole is substantially reduced when compared to $g_{I I}(r)$ mainly due to the weaker Coulomb repulsion. In stark contrast, the spinoffdiagonal pair correlation function $g_{\uparrow \downarrow}(r)$ [circles] exhibits a nontrivial behavior and strongly depends on the temperature. While being nearly flat for $T=190 \mathrm{eV}, g_{\uparrow \downarrow}(r)$ markedly increases towards $r=0$ for $T=100 \mathrm{eV}$. This increased contact probability is a direct consequence of the substantial presence of ions with a fully occupied K-shell at the lower temperature, and nicely illustrates the capability of our PIMC simulations to capture the complex interplay of ionization, thermal excitation, and electron-electron correlations. Finally, panel f) shows corresponding results for the ion-ion [squares] and electron-ion [crosses] static structure factor (SSF). These contain important information about the generalized form factor and Rayleigh weight, which are key properties in the interpretation of XRTS experiments (11) and a gamut of other applications.

![](https://cdn.mathpix.com/cropped/2024_06_04_503fd8a2efcadbf2d272g-07.jpg?height=1368&width=1594&top_left_y=416&top_left_x=255)
a)
![](https://cdn.mathpix.com/cropped/2024_06_04_503fd8a2efcadbf2d272g-07.jpg?height=844&width=1556&top_left_y=926&top_left_x=272)

![](https://cdn.mathpix.com/cropped/2024_06_04_503fd8a2efcadbf2d272g-07.jpg?height=420&width=547&top_left_y=1351&top_left_x=1277)

Figure 2: a) Schematic illustration of our setup. The Be capsule is compressed and probed by an x-ray source (purple); the scattered photons (blue) are collected by a detector under an angle $\theta$ with respect to the incident beam. We use the PIMC method to simulate the quantum degenerate interior of the capsule, allowing for unprecedented comparisons between theory and experiment; b) XRTS spectra for $\theta=120^{\circ}$ (green) and $\theta=75^{\circ}$ (red), and the source-andinstrument function (blue dashed); c,f) PIMC results for $S_{e e}(q)$ for different densities (symbols) compared to the NIF data point (bold blue) and random phase approximation (RPA) results (dotted lines); d,g) ITCF $F_{e e}(q, \tau)$ in the $q$ - $\tau$-plane, with the colored surface and dashed blue line corresponding to PIMC simulations for $\rho=20 \mathrm{~g} / \mathrm{cc}$ and the Laplace transform of the NIF spectra; e,h) $\tau$-dependence of $F_{e e}(q, \tau)$ at the probed wavenumber. The center and bottom rows correspond to the $\theta=120^{\circ}$ and $\theta=75^{\circ}$ shots, for which we find $T=155.5 \mathrm{eV}$ and $T=190 \mathrm{eV}$ (see the vertical dotted lines in e,h), respectively (30).

Results. As a demonstration of our new PIMC capabilities, we re-analyze an XRTS experiment with strongly compressed beryllium at the National Ignition Facility (11) and repeated the experiment at a smaller scattering angle to focus more explicitly on electronic correlation effects. Fig. 21) shows an illustration of the experimental set-up using the GBar XRTS platform. 184 optical laser beams (not shown) are used for the hohlraum compression (11) of a Be capsule (yellow sphere) which is filled with a core of air. A further 8 laser beams are used to heat a zinc foil generating $8.9 \mathrm{keV}$ X-rays (31) that are used to probe the system (purple ray). By detecting the scattered intensity (blue ray) at an angle $\theta$, we get insight into the microscopic physics of the sample on a specific length scale; the same microscopic physics can be resolved by our new PIMC simulations, a snapshot of which is depicted inside the Be capsule.

The measured XRTS spectra are shown as the green and red curves in Fig. 2b) and have been obtained at scattering angles of $\theta=120^{\circ}$ (11) and $\theta=75^{\circ}$ (new). They are given by a convolution of the dynamic structure factor $S_{e e}(q, \omega)$ with the combined source-and-instrument function $R(\omega)$ [dashed blue]. Since a deconvolution to extract $S_{e e}(q, \omega)$ is unstable, we instead perform a two-sided Laplace transform (32, 33, 22)

$$
\begin{equation*}
F_{e e}(q, \tau)=\mathcal{L}\left[S_{e e}(q, \omega)\right]=\int_{-\infty}^{\infty} \mathrm{d} \omega S_{e e}(q, \omega) e^{-\hbar \omega \tau} \tag{1}
\end{equation*}
$$

the well-known convolution theorem then gives us direct access to the imaginary-time correlation function (ITCF) $F_{e e}(q, \tau)$ based on the experimental data, with $\tau \in[0, \beta]$ being the imaginary time and $\beta=1 / k_{\mathrm{B}} T$ the inverse temperature $(30)$. The ITCF contains the same information as $S_{e e}(q, \omega)$, but in a different representation (26). A particularly important application of $F_{e e}(q, \tau)$ is the model-free estimation of the temperature 32, 33), and we find $T=155.5 \mathrm{eV}$ and $T=190 \mathrm{eV}$ for $\theta=120^{\circ}$ and $\theta=75^{\circ}$, respectively.

A second advantage of Eq. (1) is that it facilitates the direct comparison of the experimental observation with our new PIMC results. As a first point, we consider the electronic static
structure factor $S_{e e}(q)=F_{e e}(q, 0)$ in Fig. 2 k) and f) for the two relevant temperatures, and the circles and crosses show PIMC results for $N_{\mathrm{Be}}=25$ and $N_{\mathrm{Be}}=10$ beryllium atoms. Evidently, no finite-size effects can be resolved within the given error bars with the possible exception of the smallest $q$ values. A particular strength of the PIMC method is that it allows us to unambiguously resolve the impact of electronic XC-effects. To highlight their importance for the description of warm dense quantum plasmas even in the high-density regime, we compare the PIMC data with the mean-field based random phase approximation (34) (dotted lines). The latter approach systematically underestimates the true $S_{e e}(q)$ and only becomes exact in the single-particle limit of large wave numbers. The blue circles correspond to $S_{e e}(q)$ extracted from the NIF data following the procedure introduced in the recent Ref. (35). They are consistent with the PIMC results for $\rho \lesssim 20 \mathrm{~g} / \mathrm{cc}$.

The full ITCF $F_{e e}(q, \tau)$ is shown in panels d) and g) in the $q-\tau$-plane, where the coloured surface shows the PIMC results for $\rho=20 \mathrm{~g} / \mathrm{cc}$, and the dashed blue lines have been obtained from the experimental data via a two-sided Laplace transform, see Supplemental Material (30). Clearly, the ITCF exhibits a rich structure that is mainly characterized by an increasing decay with $\tau$ for larger wave numbers. In fact, this $\tau$-dependence is a direct consequence of the quantum delocalization of the electrons (26,36) and would be absent in a classical system. The NIF data are in excellent agreement with our PIMC simulations over the entire $\tau$-range. This can be seen particularly well in panels e) and $\mathrm{h}$ ), where we show the ITCF for the fixed values of $q$ probed in the experiment. We find a more pronounced decay of $F_{e e}(q, \tau)$ with increasing $\tau$ for larger $q$. A second effect is driven by the different temperatures of these separate NIF shots, as a higher temperature leads to a reduction of quantum delocalization and, therefore, a reduced $\tau$-decay. The observed agreement between the PIMC results and the experimental data for different $q$ and temperature is thus nontrivial and constitutes a remarkable level of agreement and consistency between theory and experiment.

![](https://cdn.mathpix.com/cropped/2024_06_04_503fd8a2efcadbf2d272g-10.jpg?height=645&width=1502&top_left_y=401&top_left_x=298)

![](https://cdn.mathpix.com/cropped/2024_06_04_503fd8a2efcadbf2d272g-10.jpg?height=623&width=729&top_left_y=415&top_left_x=313)

![](https://cdn.mathpix.com/cropped/2024_06_04_503fd8a2efcadbf2d272g-10.jpg?height=585&width=708&top_left_y=415&top_left_x=1077)
b)

Figure 3: a) Extracting the ratio of elastic (blue) and inelastic (red) contributions $I_{\mathrm{el}} / I_{\text {inel }}$ from the XRTS measurement at $\theta=75^{\circ}\left(q=5.55 \AA^{-1}\right)$; b) wavenumber dependence of the ratio $I_{\mathrm{el}} / I_{\text {inel }}$. Solid (dotted) lines: PIMC results for $T=155.5 \mathrm{eV}(T=190 \mathrm{eV})$ for $\rho=$ $7.5 \mathrm{~g} / \mathrm{cc}$ (red), $\rho=20 \mathrm{~g} / \mathrm{cc}$ (green), and $\rho=30 \mathrm{~g} / \mathrm{cc}$ (yellow); blue cross and diamond: NIF measurements for $\theta=120^{\circ}$ and $\theta=75^{\circ}$.

Finally, we consider an additional observable that can be directly extracted from the experimental data: the ratio of the elastic to the inelastic contributions to the full scattering intensity $I_{\mathrm{el}} / I_{\text {inel }}$. In Fig. 3a), the two components are illustrated for the case of $\theta=75^{\circ}$. In practice, the elastic signal has the form of the source function [cf. Fig. $2 \mathrm{~b})]$, and $I_{\text {inel }}$ is given by the remainder. The ratio $I_{\mathrm{el}} / I_{\text {inel }}$ constitutes a distinct measure for the localization of the electrons around the ions on the probed length scale determined by $q$. Therefore, it is highly sensitive to system parameters such as the density, and, additionally, to the heuristic but often useful concept of an effective ionization degree $(11)$. Yet, the prediction of $I_{\mathrm{el}} / I_{\mathrm{inel}}$ from ab initio simulations requires detailed knowledge about correlations between all particle species, see the Supplemental Material (30). This requisition is beyond the capabilities of standard DFT-MD, but straightforward with our new PIMC simulations, cf. Fig. 11.

In Fig. 3b), we show our simulation results for $I_{\mathrm{el}} / I_{\text {inel }}$ for two temperatures and three densities. The comparison with the experimental data points yields excellent agreement with
$\rho=20 \mathrm{~g} / \mathrm{cc}$ for both experiments, which is fully consistent with the independent analysis of $S_{\text {ee }}(q)$ presented in Fig. 2. Moreover, Fig. 3p) further substantiates the difference in temperature in the two separate NIF shots that we have observed from the model-free ITCF method (30). This nicely illustrates the unprecedented degree of consistency in the analysis of XRTS signals facilitated by our new simulation capabilities.

Discussion. We have presented a novel framework for the highly accurate ab initio PIMC simulation of warm dense quantum plasmas, treating electrons and ions on the same level. As an application, we have analyzed XRTS measurements of strongly compressed Be using existing data (11) as well as a new data set that probes larger length scales where electronic XC-effects are more important. Due to their unique access to electronic correlation functions, our PIMC simulations have allowed us to independently analyze various aspects of the XRTS signal such as the ITCF $F_{e e}(q, \tau)$ and the ratio of elastic to inelastic contributions. We have thus demonstrated a remarkable consistency between simulation and experiment without the need for any empirical parameters.

Our PIMC simulations accurately capture phenomena that manifest over distinctly different length scales due to the simulation of potentially hundreds of electrons and nuclei (29). This is particularly important for upcoming XRTS measurements with smaller scattering angles, and for the description of properties that can be probed in the optical limit such as electric conductivity and reflectivity. A key strength is our capability to resolve any type of manyparticle correlation function between either electrons or ions. This is in stark contrast with standard DFT-MD simulations, where the computation of electronic correlation functions is not possible even if the exact XC functional were known. Moreover, our PIMC simulations are not confined to two-particle correlation functions and linear response properties such as the dynamic structure factor probed in XRTS (22).

Our highly accurate PIMC results will spark a host of developments in the simulation of WDM. Most importantly, they can unambiguously benchmark the accuracy of existing DFT approaches, and provide crucial input for the construction of advanced nonlocal XC functionals (37). In addition, our results can quantify the nodal error in the restricted PIMC approach, support dynamic methods including time-dependent DFT, and test the basic underlying assumptions in widely used theoretical models.

Finally, our simulations will have a direct and profound impact on nuclear fusion and astrophysics. Due to their dependable predictive capability, they provide both key input for integrated modeling such as transport properties and the equation of state and guide the development of experimental set-ups. A case in point is given by XRTS measurements, for which we have demonstrated the capabilities of our PIMC approach to give a highly consistent interpretation of the data for strongly compressed beryllium. Having unraveled electronic correlations in warm dense quantum plasmas, we open the path to study lights elements and potentially their mixtures for the extreme conditions encountered during inertial confinement fusion implosions and within astrophysical objects. This will be a true game changer for a field that previously lacked predictive capability.

## References

1. R. G. Kraus, et al., Measuring the melting curve of iron at super-earth core conditions, Science 375, 202-205 (2022).
2. A. L. Kritcher, et al., A measurement of the equation of state of carbon envelopes of white dwarfs, Nature 584, 51-54 (2020).
3. R. E. Hanneman, H. M. Strong, F. P. Bundy, Hexagonal diamonds in meteorites: Implications, Science 155, 995-997 (1967).
4. D. Kraus, et al., Nanosecond formation of diamond and lonsdaleite by shock compression of graphite, Nature Communications 7, 10970 (2016).
5. R. Betti, O. A. Hurricane, Inertial-confinement fusion with lasers, Nature Physics 12, 435448 (2016).
6. A. B. Zylstra, et al., Burning plasma achieved in inertial fusion, Nature 601, 542-548 (2022).
7. H. Abu-Shawareb, et al., Lawson criterion for ignition exceeded in an inertial fusion experiment, Phys. Rev. Lett. 129, 075001 (2022).
8. H. Abu-Shawareb, et al., Achievement of target gain larger than unity in an inertial fusion experiment, Phys. Rev. Lett. 132, 065102 (2024).
9. D. Kraus, et al., Formation of diamonds in laser-compressed hydrocarbons at planetary interior conditions, Nature Astronomy 1, 606-611 (2017).
10. J. E. Bailey, et al., A higher-than-predicted measurement of iron opacity at solar interior temperatures, Nature 517, 56-59 (2015).
11. T. Döppner, et al., Observing the onset of pressure-driven k-shell delocalization, Nature $618,270-275$ (2023).
12. S. Malko, et al., Proton stopping measurements at low velocity in warm dense carbon, Nature Communications 13, 2893 (2022).
13. W. Cayzac, et al., Experimental discrimination of ion stopping models near the bragg peak in highly ionized matter, Nature Communications 8, 15693 (2017).
14. N. D. Mermin, Thermal properties of the inhomogeneous electron gas, Phys. Rev. 137, A1441-A1443 (1965).
15. D. M. Ceperley, Path integrals in the theory of condensed helium, Rev. Mod. Phys 67, 279 (1995).
16. M. Troyer, U. J. Wiese, Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations, Phys. Rev. Lett 94, 170201 (2005).
17. T. Dornheim, Fermion sign problem in path integral Monte Carlo simulations: Quantum dots, ultracold atoms, and warm dense matter, Phys. Rev. E 100, 023307 (2019).
18. T. Dornheim, S. Groth, M. Bonitz, The uniform electron gas at warm dense matter conditions, Phys. Reports 744, 1-86 (2018).
19. E. W. Brown, B. K. Clark, J. L. DuBois, D. M. Ceperley, Path-integral monte carlo simulation of the warm dense homogeneous electron gas, Phys. Rev. Lett. 110, 146405 (2013).
20. B. Militzer, K. P. Driver, Development of path integral monte carlo simulations with localized nodal surfaces for second-row elements, Phys. Rev. Lett. 115, 176403 (2015).
21. B. Militzer, F. González-Cataldo, S. Zhang, K. P. Driver, F. m. c. Soubiran, First-principles equation of state database for warm dense matter computation, Phys. Rev. E 103, 013203 (2021).
22. T. Dornheim, et al., Electronic density response of warm dense matter, Physics of Plasmas 30, 032705 (2023).
23. D. M. Ceperley, Fermion nodes, Journal of Statistical Physics 63, 1237-1267 (1991).
24. T. Schoof, S. Groth, J. Vorberger, M. Bonitz, Ab initio thermodynamic results for the degenerate electron gas at finite temperature, Phys. Rev. Lett. 115, 130402 (2015).
25. F. D. Malone, et al., Accurate exchange-correlation energies for the warm dense electron gas, Phys. Rev. Lett. 117, 115701 (2016).
26. T. Dornheim, Z. Moldabekov, P. Tolias, M. Böhme, J. Vorberger, Physical insights from imaginary-time density-density correlation functions, Matter Radiat. Extremes 8, 056601 (2023).
27. Y. Xiong, H. Xiong, On the thermodynamic properties of fictitious identical particles and the application to fermion sign problem, The Journal of Chemical Physics 157, 094112 (2022).
28. T. Dornheim, et al., Fermionic physics from ab initio path integral Monte Carlo simulations of fictitious identical particles, J. Chem. Phys. 159, 164113 (2023).
29. T. Dornheim, S. Schwalbe, Z. A. Moldabekov, J. Vorberger, P. Tolias, Ab initio path integral Monte Carlo simulations of the uniform electron gas on large length scales, J. Phys. Chem. Lett. 15, 1305-1313 (2024).
30. See Supplemental Material for additional details.
31. M. J. MacDonald, et al., Demonstration of a laser-driven, narrow spectral bandwidth x-ray source for collective x-ray scattering experiments, Physics of Plasmas 28, 032708 (2021).
32. T. Dornheim, et al., Accurate temperature diagnostics for matter under extreme conditions, Nature Communications 13, 7911 (2022).
33. T. Dornheim, et al., Imaginary-time correlation function thermometry: A new, highaccuracy and model-free temperature analysis technique for x-ray Thomson scattering data, Physics of Plasmas 30, 042707 (2023).
34. W. Kraeft, D. Kremp, W. Ebeling, G. Röpke, Quantum Statistics of Charged Particle Systems (Springer US, 2012).
35. T. Dornheim, et al., X-ray thomson scattering absolute intensity from the f-sum rule in the imaginary-time domain, arXiv:2305.15305 (2023).
36. T. Dornheim, J. Vorberger, Z. A. Moldabekov, M. Böhme, Analysing the dynamic structure of warm dense matter in the imaginary-time domain: theoretical models and simulations, Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 381, 20220217 (2023).
37. A. Pribram-Jones, P. E. Grabowski, K. Burke, Thermal density functional theory: Timedependent linear response and approximate functionals from the fluctuation-dissipation theorem, Phys. Rev. Lett 116, 233001 (2016).
38. G. Giuliani, G. Vignale, Quantum Theory of the Electron Liquid (Cambridge University Press, Cambridge, 2008).
39. M. P. Böhme, et al., Evidence of free-bound transitions in warm dense matter and their impact on equation-of-state measurements, arXiv:2306.17653 (2023).
40. J. Vorberger, D. O. Gericke, Ab initio approach to model x-ray diffraction in warm dense matter, Phys. Rev. E 91, 033112 (2015).
41. L. M. Fraser, et al., Finite-size effects and coulomb interactions in quantum monte carlo calculations for homogeneous systems with periodic boundary conditions, Phys. Rev. B 53, $1814-1832$ (1996).
42. E. Prodan, W. Kohn, Nearsightedness of electronic matter, Proceedings of the National Academy of Sciences 102, 11635-11638 (2005).
43. M. Böhme, Z. A. Moldabekov, J. Vorberger, T. Dornheim, Ab initio path integral monte carlo simulations of hydrogen snapshots at warm dense matter conditions, Phys. Rev. E 107, 015206 (2023).
</end of paper 1>


<paper 2>
# Ab initio path integral Monte Carlo simulations of warm dense two-component systems without fixed nodes: structural properties 

Tobias Dornheim, ${ }^{1,2, *}$ Sebastian Schwalbe,, ${ }^{1,2}$ Maximilian P. Böhme, ${ }^{1,2,3}$<br>Zhandos A. Moldabekov, ${ }^{1,2}$ Jan Vorberger, ${ }^{2}$ and Panagiotis Tolias ${ }^{4}$<br>${ }^{1}$ Center for Advanced Systems Understanding (CASUS), D-02826 Görlitz, Germany<br>${ }^{2}$ Helmholtz-Zentrum Dresden-Rossendorf (HZDR), D-01328 Dresden, Germany<br>${ }^{3}$ Technische Universität Dresden, D-01062 Dresden, Germany<br>${ }^{4}$ Space and Plasma Physics, Royal Institute of Technology (KTH), Stockholm, SE-100 44, Sweden


#### Abstract

We present extensive new $a b$ initio path integral Monte Carlo (PIMC) results for a variety of structural properties of warm dense hydrogen and beryllium. To deal with the fermion sign problem-an exponential computational bottleneck due to the antisymmetry of the electronic thermal density matrix-we employ the recently proposed [J. Chem. Phys. 157, 094112 (2022); 159, 164113 (2023)] $\xi$-extrapolation method and find excellent agreement with exact direct PIMC reference data where available. This opens up the intriguing possibility to study a gamut of properties of light elements and potentially material mixtures over a substantial part of the warm dense matter regime, with direct relevance for astrophysics, material science, and inertial confinement fusion research.


## I. INTRODUCTION

Warm dense matter (WDM) is an extreme state that is characterized by the simultaneous presence of high temperatures, pressures, and densities [1]. In nature, WDM occurs in a host of astrophysical objects, including giant planet interiors [2], brown dwarfs [3], and the outer layer of neutron stars [4]. In addition, WDM plays an important role for technological applications such as hotelectron chemistry [5], and the discovery and synthesis of exotic materials [6-8]. A particularly important example is given by inertial confinement fusion (ICF) [9], where the fuel capsule (typically a mixture of the hydrogen isotopes deuterium and tritium) has to traverse the WDM regime in a controlled way to reach ignition [10]. In fact, the recently reported [11] net energy gain from an ICF experiment has opened up the intriguing possibility to further develop ICF into an economically viable and sustainable option for the production of clean energy [12].

Yet, the rigorous theoretical description of WDM constitutes a notoriously difficult challenge $[1,13-15]$, as it must capture the complex interplay of Coulomb coupling and nonideality, quantum degeneracy and diffraction, as well as strong thermal excitations out of the ground state. From a theoretical perspective, these conditions are conveniently characterized in terms of the Wigner-Seitz radius (also known as density parameter or quantum coupling parameter) $r_{s}=(3 / 4 \pi n)^{1 / 3}$, where $n=N / \Omega$ is the electronic number density, and the degeneracy temperature $\Theta=k_{\mathrm{B}} T / E_{\mathrm{F}}$, with $E_{\mathrm{F}}$ the usual Fermi energy [16]. In the WDM regime, one has $r_{s} \sim \Theta \sim 1$ [17], which means that there are no small parameters for an expansion [1].

In this situation, thermal density functional theory (DFT) [18] has emerged as the de-facto work horse of WDM theory as it combines a generally manageable computation cost with an often acceptable level of accuracy.[^0]

At the same time, it is important to note that DFT crucially relies on external input i) to provide the required exchange-correlation (XC) functional, ii) as a benchmark for its inherent approximations. At ambient conditions where the electrons are in their ground state, the availability of highly accurate $a b$ initio quantum Monte Carlo (QMC) results for the uniform electron gas (UEG) [1922] was of paramount importance to facilitate the arguably unrivaled success of DFT with respect to the description of real materials [23]. Yet, it is easy to see that thermal DFT simulations of WDM require as an input a thermal XC-functional that depends on both density and the temperature $[1,24,25]$, the construction of which, in turn, must be based on thermal QMC simulations at WDM conditions $[26]$.

In this context, the $a b$ initio path integral Monte Carlo (PIMC) method [27-29] constitutes a key concept. On the one hand, PIMC is, in principle, capable of providing an exact solution to a given quantum many-body problem of interest for arbitrary temperatures; this has given insights into important phenomena such as superfluidity [27, 30, 31] and Bose-Einstein condensation [32, 33]. On the other hand, PIMC simulations of Fermi systems such as the electrons in WDM are afflicted with the notorious fermion sign problem (FSP) [34-36]. It leads to an exponential increase in the required compute time with increasing the system size $N$ or decreasing the temperature $T$, and has prevented PIMC simulations over a substantial part of the WDM regime.

This unsatisfactory situation has sparked a surge of developments in the field of fermionic $\mathrm{QMC}$ simulations targeting WDM [37-45], see Refs. [13, 14, 46] and references therein. These efforts have culminated in the first accurate parametrizations of the XC-free energy of the warm dense UEG [13, 47-50], which can be used as input for thermal DFT simulations of WDM [24,51-56] on the level of the local density approximation. While being an important milestone, the development of highly accurate PIMC simulations of real WDM systems where the posi-
tively charged nuclei are not approximated by a homogeneous background remains of the utmost importance for many reasons, e.g.: i) it is important to benchmark different XC-functionals for thermal DFT simulations in realistic situations; ii) PIMC results can be used to construct more sophisticated thermal XC-functionals, e.g. via the fluctuation-dissipation theorem [57]; iii) PIMC gives one straightforward access to the imaginary-time densitydensity correlation function (ITCF) $F_{e e}(\mathbf{q}, \tau)$, which is of key importance for the interpretation of X-ray Thomson scattering (XRTS) experiments with WDM [14, 58-62]; iv) while DFT is an effective single-electron theory, PIMC allows one to compute many-body correlation functions and related nonlinear effects $[14,63-67]$.

As a consequence, a number of strategies to attenuate the FSP in PIMC simulations have been discussed in the literature $[13,37,39,45,68-78]$. Note that complementary QMC methods like configuration PIMC [13, 38, 44, 79-81] and density matrix QMC [41, 42, 82] lie beyond the scope of the present work and have been covered elsewhere. Three decades ago, Ceperley [72] suggested a reformulation of the PIMC method, where the sampling of any negative contributions can be avoided by prohibiting paths that cross the nodal surfaces of the thermal density matrix. This fixed-node approximation, also known as restricted PIMC (RPIMC) in the literature, is both formally exact and sign-problem free. Unfortunately, the exact nodal surface of an interacting many-fermion system is generally a-priori unknown, and one has to rely on de-facto uncontrolled approximations in practice. For the warm dense UEG, Schoof et al. [38] have reported systematic errors of $\sim 10 \%$ in the XC-energy at high density $\left(r_{s}=1\right)$, whereas Dornheim et al. [83] have found better agreement for the momentum distribution for $r_{s} \geq 2$. In addition, the nodal restrictions on which RPIMC is based break the usual imaginary-time translation invariance and, thus, prevent the straightforward estimation of imaginary-time correlation functions. In spite of these shortcomings, the RPIMC method has been successfully applied by Militzer and coworkers $[73,74,84,85]$ to a variety of WDM systems, and their results form the basis for an extensive equation-of-state database [86].

A different line of thought has been pursued by Takahashi and Imada [28], who have suggested to use inherently anti-symmetric imaginary-time propagators, i.e., determinants. This leads to the blocking (grouping together) of positive and negative contributions to the partition function into a single term, thereby alleviating the sign problem. Similar considerations have been used by Filinov et al. for PIMC simulations of the UEG [87], warm dense hydrogen [88], and exotic quark-gluon plasmas [89]. In practice, this strategy works well if the number of imaginary-time propagators $P$ is small enough so that the thermal wavelength of a single imaginary-time slice $\lambda_{\epsilon}=\sqrt{2 \pi \epsilon}$, with $\epsilon=\beta / P$, is comparable to the average interparticle distance $\bar{r}$. The key point is thus to combine the determinant with a high-order factorization of the density matrix [90], which allows for suffi- cient accuracy even for very small $P$. This is the basic idea of the permutation blocking PIMC (PB-PIMC) method $[13,39,45,69,91,92]$, which has very recently been applied to the simulation of warm dense hydrogen by Filinov and Bonitz [93]. While being arguably less uncontrolled than the RPIMC method, the PB-PIMC idea has three main bottlenecks: i) even though the FSP is significantly attenuated, the method still scales exponentially with $\beta$ and $N$; ii) the convergence with $P$ is often difficult to ensure, especially at low $T$ where this would be most important; iii) it is difficult to resolve imaginarytime properties due to the necessarily small number of imaginary-time slices.

A third route has been recently suggested by Xiong and Xiong [75, 76] based on the path integral molecular dynamics simulation of fictitious identical particles that are defined by the continuous spin-variable $\xi \in[-1,1]$ [cf. Eq. (2) below]. To be more specific, they have proposed to carry out simulations in the sign-problem free domain of $\xi \geq 0$, and to subsequently extrapolate to the fermionic limit of $\xi=-1$. This $\xi$-extrapolation method has subsequently been adapted to PIMC simulations of the warm dense UEG by Dornheim et al. [77, 78], who have found that it works remarkably well for weak to moderate levels of quantum degeneracy. Although such an extrapolation is empirical, it combines a number of strong advantages in practice. First and foremost, only effects due to quantum statistics have to be extrapolated. These are known to be local in the case of fermions [94], which means that it is possible to verify the applicability (or lack thereof) of the $\xi$-extrapolation method for relatively small systems (e.g. $N \sim 4, \ldots, 14$ ) before applying it to larger numbers of particles where no direct check is possible [77]. Second, the method has no sign-problem, which allows one to simulate very large numbers of electrons [78]. Finally, it gives one access to all observables that can be computed with direct PIMC, including the ITCF. This has very recently allowed Dornheim et al. [95] to compare extensive PIMC simulations of warm dense Be to XRTS experiments carried out at the National Ignition Facility (NIF) [96], resulting in an unprecedented consistency between theory and experiment without the need for any empirical parameters.

In the present work, we report a detailed study of the application of the $\xi$-extrapolation method to warm dense two-component plasmas, focusing on various structural characteristics of hydrogen and beryllium. The paper is organized as follows. In Sec. II, we introduce the relevant theoretical background, including a definition of the all-electron Hamiltonian governing the entire twocomponent system (II A), a brief introduction to the $a b$ initio PIMC method (II B) and a subsequent overview of the $\xi$-extrapolation method (II C). In Sec. III, we present our extensive new simulation results, starting with the fermion sign problem (III A); interestingly, it is substantially more severe for two-component plasmas than for the UEG at the same conditions, reflecting the more complex physics in a real WDM system. In Secs. III B
and III C, we present a detailed analysis of hydrogen at the electronic Fermi temperature $(\Theta=1)$ at a metallic $\left(r_{s}=2, \rho=0.34 \mathrm{~g} / \mathrm{cc}, T=12.53 \mathrm{eV}\right)$ and solid $\left(r_{s}=3.23\right.$, $\rho=0.08 \mathrm{~g} / \mathrm{cc}, T=4.80 \mathrm{eV}$ ) density. In Sec. III D, we consider the substantially more complex case of strongly compressed ( $\left.r_{s}=0.93, \rho=7.49 \mathrm{~g} / \mathrm{cc}\right)$ Be at $T=100 \mathrm{eV}$ $(\Theta=1.73)$, which is relevant e.g. for experiments at the NIF [95-97]. Remarkably, the $\xi$-extrapolation method is even capable of reproducing the correct interplay of XC-effects with double occupation of the atomic K-shell as they manifest in observables such as the spin-resolved pair correlation function. Finally, we consider the spatially resolved electronic density in the external potential of a fixed ion snapshot in Sec. IIIE. The paper is con- cluded by a summary and outlook in Sec. IV.

## II. THEORY

## A. Hamiltonian

Throughout this work, we restrict ourselves to the fully unpolarized case where $N_{\uparrow}=N_{\downarrow}=N / 2$ with $N$ being the total number of electrons in the system. Moreover, we consider effectively charge neutral systems where $N=Z N_{\text {atom }}$ with $Z$ being the nuclear charge and $N_{\text {atom }}$ the total number of nuclei. The corresponding Hamiltonian governing the behaviour of the thus defined twocomponent plasma then reads

$$
\begin{equation*}
\hat{H}=-\frac{1}{2} \sum_{l=1}^{N} \nabla_{l}^{2}-\frac{1}{2 m_{n}} \sum_{l=1}^{N_{\text {atom }}} \nabla_{l}^{2}+\sum_{l<k}^{N} W_{\mathrm{E}}\left(\hat{\mathbf{r}}_{l}, \hat{\mathbf{r}}_{k}\right)+Z^{2} \sum_{l<k}^{N_{\text {atom }}} W_{\mathrm{E}}\left(\hat{\mathbf{I}}_{l}, \hat{\mathbf{I}}_{k}\right)-Z \sum_{k=1}^{N} \sum_{l=1}^{N_{\text {atom }}} W_{\mathrm{E}}\left(\hat{\mathbf{I}}_{l}, \hat{\mathbf{r}}_{k}\right) \tag{1}
\end{equation*}
$$

where $\hat{\mathbf{r}}$ and $\hat{\mathbf{I}}$ denote electron and nucleus position operators, and we assume Hartree atomic units (i.e., $m_{e}=1$ ) throughout this work. The pair interaction is given by the usual Ewald potential, where we follow the convention given by Fraser et al. [98].

## B. Path integral Monte Carlo

The $a b$ initio PIMC method constitutes one of the most successful tools in statistical physics and quantum chemistry. Since more detailed introductions to PIMC [27-29] and its algorithmic implementation [83, 99] have been presented in the literature, we restrict ourselves to a brief summary of the main ideas. As a starting point, we consider the canonical (i.e., number of particles $N$, volume $\Omega$, and inverse temperature $\beta=1 / k_{\mathrm{B}} T$ are fixed) partition function in coordinate representation,

$$
\begin{equation*}
Z_{N, \Omega, \beta}=\frac{1}{N_{\uparrow}!N_{\downarrow}!} \sum_{\sigma_{N_{\uparrow}} \in S_{N_{\uparrow}}} \sum_{\sigma_{N_{\downarrow}} \in S_{N_{\downarrow}}} \xi^{N_{p p}} \int \mathrm{d} \mathbf{R}\left\langle\mathbf{R}\left|e^{-\beta \hat{H}}\right| \hat{\pi}_{\sigma_{N_{\uparrow}}} \hat{\pi}_{\sigma_{N_{\downarrow}}} \mathbf{R}\right\rangle \tag{2}
\end{equation*}
$$

where $\mathbf{R}$ contains the coordinates of all electrons and nuclei. The double sum over all possible permutations of coordinates of the spin-up and spin-down electrons is required to properly realize the fermionic antisymmetry of the thermal density matrix, where the exponent $N_{p p}$ corresponds to the number of pair permutations that is required to realize a particular combination of permutation elements $\sigma_{N_{\uparrow}}$ and $\sigma_{N_{\downarrow}}$. In general, the cases $\xi=1$, $\xi=0$, and $\xi=-1$ correspond to Bose-, Boltzmann-, and Fermi-statistics, with the latter applying to the electrons studied in the present work. We note that we treat the nuclei as distinguishable Boltzmann particles (i.e., boltzmannons), which is appropriate for the conditions studied in the present work. Further note that this does still take into account nuclear quantum effects due to the finite extension of the nucleonic paths, which are however small ( 0.1\%) in the warm dense matter regime even for hydrogen.

The problem with Eq. (2) concerns the evaluation of the matrix elements of the density operator $\hat{\rho}=e^{-\beta \hat{H}}$ that is not directly possible in practice, because the potential and kinetic contributions to the full Hamiltonian $\hat{H}=\hat{V}+\hat{K}$ do not commute. The usual workaround is to evoke the exact semi-group property of the density operator combined with a Trotter decomposition [100] of the latter, leading to the evaluation of $P$ density matrices at a $P$-times higher temperature. In the case of electron-ion systems, an additional obstacle is given by the diverging Coulomb attraction between electrons and ions at short distances; this prevents a straightforward application of the Trotter formula, which only holds for potentials that are bounded from below. We over-
come this problem by using the well-known pair approximation [27, 101-103] that is based on the exact solution of the isolated electron-ion two-body problem, and utilize the implementation presented in Ref. [103]. In essence, the quantum many-body problem defined by Eq. (2) has then been mapped onto an ensemble of quasiclassical ring polymers with $P$ segments [104]; these are the eponymous paths of the PIMC method. Alternatively, one can say that each quantum particle is now represented by an entire path of length $\tau=\beta$ along the imaginary time $t=-i \hbar \tau$ with $P$ discrete imaginarytime steps. Therefore, PIMC gives one straightforward access to a number of imaginary-time correlation functions $[66,99,105,106]$, with the density-density correlator $F_{a b}(\mathbf{q}, \tau)=\left\langle\hat{n}_{a}(\mathbf{q}, \tau) \hat{n}_{b}(-\mathbf{q}, 0)\right\rangle$ between species $a$ and $b$ being a particularly relevant example for WDM research [58-60, 107-109]. The PIMC formulation becomes exact in the limit of $P \rightarrow \infty$, and the convergence with $P$ has been carefully checked. We find that $P=200$ proves sufficient for all cases studied here.

The basic idea of the PIMC method is to randomly generate all possible path configurations $\mathbf{X}$ (where the meta variable $\mathbf{X}$ contains both the coordinates of electrons and nuclei) using an implementation of the celebrated Metropolis algorithm [110]. We note that this also requires the sampling of different permutation topologies, which can be accomplished efficiently via the worm algorithm that was introduced by Boninsegni et al. [99, 111]. In practice, we use the extended ensemble algorithm from Ref. [83] which is implemented in the ISHTAR code [112].

## C. The $\xi$-extrapolation method

In the case of fermions, the factor of $(-1)^{N_{p p}}$ leads to a cancellation of positive and negative terms, which is the root cause of the notorious fermion sign problem [34-36]; it results in an exponential increase in the compute time with $N$ and $\beta$. In other words, the signal-to-noise ratio of the direct PIMC method vanishes for large systems, and at low temperatures. As a partial remedy of this bottleneck, Xiong and Xiong [75] have suggested to carry out path integral molecular dynamics simulations of fictitious identical particles where $\xi \in[-1,1]$ is treated as a continuous variable. It is straightforward to carry out highly accurate simulations in the sign-problem free domain of $\xi \in[0,1]$; one can then extrapolate to the fermionic limit of $\xi=-1$ using the empirical relation

$$
\begin{equation*}
A(\xi)=a_{0}+a_{1} \xi+a_{2} \xi^{2} \tag{3}
\end{equation*}
$$

For completeness, we note that an alternative extrapolation method has been introduced very recently [76], but the possibility of extending it to observables beyond the total energy remains a subject for dedicated future works.

Subsequently, Dornheim et al. [77] have adapted the $\xi$-extrapolation approach for PIMC simulations of the warm dense UEG, and found that is works remarkably

![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-04.jpg?height=797&width=789&top_left_y=176&top_left_x=1123)

FIG. 1. Snapshot of an all-electron PIMC simulation of $N_{\text {atom }}=25$ Be atoms at $r_{s}=0.93(\rho=7.49 \mathrm{~g} / \mathrm{cc})$ and $\Theta=1.73$ $(T=100 \mathrm{eV})$. The green orbs depict the approximately pointlike nuclei, and the red-blue paths represent the quantum degenerate (i.e., smeared out) electrons.

well for weak to moderate quantum degeneracy. A particular practical advantage of this method is that it can be rigorously verified for small systems $[N=\mathcal{O}(1)-\mathcal{O}(10)]$, where comparison with exact direct PIMC calculations is feasible. Then, a breakdown of the $\xi$-extrapolation [i.e., Eq. (3)] for larger systems at the same conditions is rendered highly unlikely due to the well-known local nature of quantum statistics effects for fermions [94]. Other effects, such as the interplay of long-range Coulomb coupling with the quantum delocalization of the electrons is still fully taken into account by the PIMC simulations in the sign-problem free domain of $\xi \in[0,1]$ for larger systems. This is illustrated in Fig. 1, where we show a snapshot of a PIMC simulation of $N_{\text {atom }}=25$ Be atoms at $r_{s}=0.93(\rho=7.49 \mathrm{~g} / \mathrm{cc})$ and $\Theta=1.73(T=100 \mathrm{eV})$. The green orbs represent the Be nuclei, which are indistinguishable from classical point particles at these conditions, even though this assumption is not hardwired into our set-up, see Sec. II B above. The red-blue paths represent the quantum degenerate electrons, with their extension being proportional to the thermal wavelength $\lambda_{\beta}=\sqrt{2 \pi \beta}$. Thus, effects such as quantum diffraction are covered on all length scales, which has recently allowed four of us to present unprecedented simulations of the warm dense UEG with up to $N=1000$ electrons [78]. Furthermore, the $\xi$-extrapolation method has been used in Ref. [95] to carry out extensive PIMC simulations of strongly compressed $\mathrm{Be}(\rho=7.5-30 \mathrm{~g} / \mathrm{cc})$, resulting in excellent agreement with XRTS measurements taken at the NIF [96]. Here, we present a much more detailed tech-

![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-05.jpg?height=770&width=770&top_left_y=179&top_left_x=214)

FIG. 2. PIMC results for the dependence of the average sign $S$ on the system size $N$ at the electronic Fermi temperature, $\Theta=1$. Blue diamonds (green crosses): UEG at $r_{s}=2\left(r_{s}=\right.$ 3.23); red circles (black stars): hydrogen at $r_{s}=2$ ( $r_{s}=3.23$ ). Dashed black: exponential fit to the data at $r_{s}=3.23$. The UEG data for $r_{s}=2$ are partially taken from Ref. [114].

nical analysis of this approach, and apply it to hydrogen and beryllium for a set of different conditions.

## III. RESULTS

All PIMC results that are presented in this work are freely available in an online repository, see Ref. [113].

## A. Fermion sign problem

The FSP constitutes the main bottleneck for PIMC simulations of WDM systems. It is a direct consequence of the cancellation of positive and negative contributions to the partition function Eq. (2) due to the antisymmetry of the thermal density matrix under the exchange of particles. This cancellation is conveniently characterized by the average sign $S$ that is inversely proportional to the relative uncertainty of a given observable [34]. In Fig. 2, we show the dependence of $S$ on the number of electrons $N$ based on PIMC calculations. The blue diamonds and yellow squares have been obtained for the UEG at $\Theta=1$ and $r_{s}=2$ and $r_{s}=3.23$, respectively. For an ideal Fermi gas, the degree of quantum degeneracy would be exclusively a function of $\Theta$ and the results independent of the density. For the UEG, $r_{s}$ serves as a quantum coupling parameter [16]. The stronger coupling at the solid density thus leads to an effective separation of the electrons within the PIMC simulation, leading to a decreased probability for the formation of permutation cycles [115]. In other words, a sparser electron gas is less quantum degenerate than a dense electron gas, resulting in the well-known monotonic increase of $S$ with $r_{s}$ for the UEG. In addition, we find the expected exponential decrease of the sign with the number of electrons that is well reproduced by the exponential fit to the yellow squares, and which is ultimately responsible for the exponentially vanishing signal-to-noise ratio for direct PIMC simulations of fermions. In practice, simulations are generally feasible for $S \sim \mathcal{O}\left(10^{-1}\right)$.

Let us next consider the red circles, which show PIMC results for hydrogen at $r_{s}=2$ (and $\Theta=1$ ). Evidently, the average sign is systematically lower compared to the UEG at the same conditions. This is a direct consequence of the presence of the protons, leading to local inhomogeneities. To be more specific, electrons tend, on average, to cluster around the protons; this even holds at conditions where the majority of electrons can be considered as effectively unbound. The reduced average distance between the electrons then leads to an increased sampling of permutation cycles and, therefore, a reduced average sign compared to the UEG.

Finally, the green crosses show results for hydrogen at $r_{s}=3.23$. Interestingly, we find the opposite trend compared to the UEG: the average sign decreases with increasing $r_{s}$. In other words, quantum degeneracy effects are even somewhat more important at lower density compared to the high-density case. This is a consequence of the more complex physics in real two-component plasmas compared to the UEG, leading to two competing trends for hydrogen. On the one hand, increasing the density compresses the electronic component, making exchange effects more important. On the other hand, reducing the density leads to the formation of $\mathrm{H}_{2}$ molecules, where the two electrons are in very close proximity. In fact, the PIMC sampling by itself cannot directly distinguish between a spin-polarized or spin-unpolarized $\mathrm{H}_{2}$ molecule. As we will see below, the correct predominance of the unpolarized case is exclusively a consequence of the fermionic antisymmetry that is realized by the cancellation of positive and negative contributions in PIMC, i.e., a subsequent re-weighting of the actually sampled set of configurations. In practice, this crucial effect is nicely captured by the $\xi$-extrapolation method, making it reliable beyond the comparably simple UEG model system.

## B. Hydrogen: metallic density

In Fig. 3, we consider hydrogen at $r_{s}=2(\rho=0.34 \mathrm{~g} / \mathrm{cc})$ and $\Theta=1(T=12.53 \mathrm{eV})$. These conditions are at the heart of the WDM regime $[1,13]$ and are realized, for example, on the compression path of a fuel capsule in an ICF experiment at the NIF [116]. From a physical perspective, hydrogen is expected to be strongly ionized in this regime $[85,93,117]$, which means that the electrons can be expected to behave similarly to the UEG at the
![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-06.jpg?height=1190&width=1610&top_left_y=169&top_left_x=239)

FIG. 3. Ab initio PIMC results for hydrogen with $N=14, r_{s}=2$, and $\Theta=1$. a) the electron-electron SSF $S_{e e}(\mathbf{q})$, b) thermal electron-electron structure factor $F_{e e}(\mathbf{q}, \beta / 2), \mathbf{c}$ ) electron-proton SSF $S_{e p}(\mathbf{q})$, d) proton-proton SSF $S_{p p}(\mathbf{q})$. Green crosses: direct (exact) PIMC results for $\xi=-1$; red circles: $\xi$-extrapolated results [Eq. (3)]; grey area: FSP free domain of $\xi \in[0,1]$.

![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-06.jpg?height=561&width=767&top_left_y=1606&top_left_x=213)

FIG. 4. The $\xi$-dependence of the electron-electron SSF $S_{e e}(\mathbf{q})$ for three wavenumbers and $N=14, r_{s}=2, \Theta=1$. Symbols: PIMC data for $q=1.53 \AA^{-1}$ (black stars), $q=3.06 \AA^{-1}$ (blue diamonds), and $q=4,59 \AA^{-1}$ (green crosses). Solid red lines: fits according to Eq. (3) based on PIMC data in the FSP free domain of $\xi \in[0,1]$. same conditions. Given the excellent performance of the $\xi$-extrapolation method for the warm dense UEG [77, 78], this regime constitutes a logical starting point for the present investigation.

In panel a), we show the electron-electron static structure factor (SSF) $S_{e e}(\mathbf{q})$ as a function of the wave number $q$ for $N=14$. The green crosses correspond to direct PIMC results for the fermionic limit of $\xi=-1$; in this case, the simulations are challenging due to the FSP, but still feasible, and we find an average sign [34] of $S=0.08466(14)$. The red circles have been computed by fitting Eq. (3) to the sign-problem free domain of $\xi \in[0,1]$ (shaded grey area) and are in excellent agreement with the exact results for all $q$. Interestingly, the SSF exhibits a more pronounced structure in the bosonic limit of $\xi=1$, which is likely a consequence of the effective attraction of two identical bosons reported in earlier works [118, 119].

In Fig. 4 , we show the $\xi$-dependence of $S_{e e}(\mathbf{q})$ for three selected wavenumbers. Specifically, the symbols show our exact direct PIMC results, which are available for all $\xi$ at these parameters, and the solid red lines fits via Eq. (3) that have been obtained exclusively based on input data
![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-07.jpg?height=1256&width=1594&top_left_y=168&top_left_x=255)

FIG. 5. Ab initio PIMC results for hydrogen with $N=14, r_{s}=2$, and $\Theta=1$. a) the spin up-down PCF $g_{u d}(r)$, b) spin up-up PCF $\left.g_{u u}(r), \mathrm{c}\right)$ rescaled electron-proton PCF $\left.g_{e p}(r), \mathrm{d}\right)$ proton-proton PCF $g_{p p}(r)$. Green crosses: direct (exact) PIMC results for $\xi=-1$; red circles: $\xi$-extrapolated results [Eq. (3)]; grey area: FSP free domain of $\xi \in[0,1]$. The dotted blue line is panel c) has been computed from the ground-state wave function of an isolated hydrogen atom, and has been rescaled arbitrarily as a guide to the eye.

from the sign-problem free domain of $\xi \in[0,1]$. The $\xi$ dependence is most pronounced for the smallest depicted $q$-value, and Eq. (3) nicely reproduces the PIMC results over the entire $\xi$-range in all cases, as it is expected.

In Fig. 3b), we analyze the thermal structure factor [120], defined by the imaginary-time density-density correlation function evaluated at its $\tau=\beta / 2$ minimum, i.e., $F_{e e}(\mathbf{q}, \beta / 2)$. From a theoretical perspective, the ITCF can be expected to depend even more strongly on quantum effects in general, and quantum statistics in particular, and thus constitutes a potentially more challenging case compared to the static $S_{e e}(\mathbf{q})=F_{e e}(\mathbf{q}, 0)$. Nevertheless, we observe that the $\xi$-extrapolation method is capable of giving excellent results for the thermal structure factor over the entire wavenumber range, just as in the previously investigated case of the UEG [77].

Let us next focus more explicitly on the effect of the nuclei (i.e., protons). To this end, we show the electronproton SSF $S_{e p}(\mathbf{q})$ and proton-proton SSF $S_{p p}(\mathbf{q})$ in panels c) and d). First and foremost, we find the same excellent agreement between the exact, direct PIMC results and the $\xi$-extrapolation results in both cases for all $q$. Second, the impact of quantum statistics is comparably reduced in particular for $S_{p p}(\mathbf{q})$, which is very similar to $S_{e e}(\mathbf{q})$ in the fermionic limit, but qualitatively differs substantially in the bosonic limit of $\xi=1$. Such direct insights into the importance of quantum degeneracy effects on different observables are a nice side effect of the $\xi$-extrapolation method.

To focus more closely on such effects, we investigate the spin-resolved electron-electron pair correlation functions (PCFs) $g_{u d}(r)$ and $g_{u u}(r)$ in Fig. 5a) and b), respectively. In the spin-offdiagonal case, hardly any effects of quantum statistics can be resolved. While somewhat expected, this is still different from $S_{p p}(\mathbf{q})$, for which quantum statistics prove to be important even though the protons themselves are effectively distinguishable, see Sec. II B above. In stark contrast, the behaviour of the spin-diagonal PCF is predominantly shaped by quantum statistics, in particular for small separations $r \rightarrow 0$. As
![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-08.jpg?height=656&width=1610&top_left_y=173&top_left_x=255)

FIG. 6. PIMC results for the electronic static structure factor $S_{e e}(\mathbf{q})$ at $r_{s}=2$ and $\Theta=1$ for $N=14$ (diamonds), $N=32$ (crosses) and $N=100$ (circles). Left: impact of quantum statistics, with green, blue and red symbols showing results for $\xi=-1$ (extrapolated), $\xi=0$, and $\xi=1$; the yellow squares show direct PIMC results for $N=14$ and $\xi=-1$. Right: comparing PIMC results for hydrogen to UEG results at the same parameters computed via the ESA [121, 122].

mentioned above, bosons tend to cluster around each other, resulting in a large contact probability $g_{u u}^{\xi=1}(0)$, and a nontrivial maximum around $r=0.4 \AA$. For distinguishable Boltzmann quantum particles, there is still a finite contact probability of $g_{u u}^{\xi=0}(0) \approx 0.5$, but the bosonic maximum for small but finite $r$ disappears. In the fermionic limit, the contact probability completely vanishes due to the Pauli exclusion principle. It is striking that the $\xi$-extrapolation method very accurately captures these stark qualitative differences for all $r$, and the red circles nicely recover the exact direct PIMC results that are shown by the green crosses. For completeness, we note that the on-top PCF $g(0)$ constitutes a very important property even for the UEG [121-124]. The applicability of the $\xi$-extrapolation method thus opens up the possibility to study $g_{e e}(0)$ and to reduce finitesize effects [123] and the possible impact of nodal errors in the case of restricted PIMC [37] by simulating larger system sizes as demonstrated in the recent Ref. [78].

In Fig. 5c) we show the electron-proton PCF $g_{e p}(r)$ [rescaled by a factor of $r^{2}$, which contains information about the electronic localization around the protons. In an atomic system, one would expect a maximum around the Bohr radius of $r=0.529 \AA$ [93], see the dotted blue curve showing the probability density of an electron around an isolated proton at the hydrogen ground state. For this observable, quantum statistics effects are mostly restricted around intermediate distances, while the contact range between an electron and a proton is mostly unaffected. Finally, we show the proton-proton PCF $g_{p p}(r)$ in Fig. $\left.5 \mathrm{~d}\right)$. It is largely featureless, except for a pronounced exchange-correlation hole around $r=0$. Interestingly, fermionic exchange effects play a more prominent role compared to $g_{e p}(r)$, and they are perfectly captured by the $\xi$-extrapolation at these conditions.
Let us conclude our analysis of the metallic density case with an investigation of finite-size effects. To this end, we show the electronic SSF $S_{e e}(\mathbf{q})$ in Fig. 6 for $N=14$ (diamonds), $N=32$ (crosses), and $N=100$ (circles) hydrogen atoms. The red, blue, and green symbols in the left panel correspond to $\xi=1, \xi=0$, and $\xi=-1$ (extrapolated), respectively; the yellow squares show the corresponding direct PIMC results for $\xi=-1$ which are available only for $N=14$. Apparently, no dependence on the system size can be resolved for the fermionic limit of prime interest for the present work. This is consistent both with previous findings for the UEG model [13, 40, 125-128], and also with the wellknown principle of electronic nearsightedness [94]. Similarly, we cannot resolve any clear dependence on $N$ in the case of $\xi=0$ despite the smaller error bars. Interestingly, this situation somewhat changes for the bosonic case of $\xi=1$, where e.g. the results for $N=14$ exhibit small differences compared to the other data. Heuristically, this can be attributed to a breakdown of nearsightedness for bosons, which, in the extreme case, are known for exhibiting off-diagonal long-range order e.g. in the case of superfluidity [129]. It is important to note that the $\xi$-extrapolation still gives the correct fermionic limit despite the larger finite-size effects in the $\xi>0$ data, as it becomes evident from the excellent agreement between the green diamonds and yellow squares.

The right panel of Fig. 6 shows a comparison of our new PIMC results for hydrogen with the UEG model at the same conditions; the latter has been computed from the effective static approximation (ESA) [121, 122], which is known to be quasi-exact in this regime. For large $q$, $S_{e e}(\mathbf{q})$ approaches the single-particle limit for both hydrogen and the UEG, and they agree for $q \gtrsim 3 \AA^{-1}$. In the long wavelength limit of $q \rightarrow 0, S_{e e}(\mathbf{q})$ is described by a
![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-09.jpg?height=1188&width=1594&top_left_y=170&top_left_x=255)

FIG. 7. Ab initio PIMC results for hydrogen with $N=14, r_{s}=3.23$, and $\Theta=1$. a) electron-electron SSF $S_{e e}(\mathbf{q})$, b) thermal electron-electron structure factor $F_{e e}(\mathbf{q}, \beta / 2)$, c) electron-proton SSF $S_{e p}(\mathbf{q})$, d) proton-proton SSF $S_{p p}(\mathbf{q})$. Green crosses: direct (exact) PIMC results for $\xi=-1$; red circles: $\xi$-extrapolated results [Eq. (3)]; grey area: FSP free domain of $\xi \in[0,1]$.

parabola vanishing for $q=0$; this is a well-known consequence of perfect screening in the UEG [13, 16, 130]. In contrast, $S_{e e}(\mathbf{q})$ attains a finite value for real electronion systems that is governed by the compressibility sumrule [131]. This can be explained by considering the definition of $S_{e e}(\mathbf{q})$ as the normalization of the dynamic structure factor,

$$
\begin{equation*}
S_{e e}(\mathbf{q})=\int_{-\infty}^{\infty} \mathrm{d} \omega S_{e e}(\mathbf{q}, \omega) \tag{4}
\end{equation*}
$$

For the UEG, $S_{e e}(\mathbf{q}, \omega)$ consists of a single (collective) plasmon peak for small $q$. For hydrogen, this free electron gas feature is complemented by a) a contribution due to transitions between bound and free states [61] and b), more importantly in this context, a quasi-elastic feature due to effectively bound electrons and the screening cloud of free electrons $[132,133]$. While the plasmon weight vanishes for small $q$, this trend does not hold for the other contributions in the case of hydrogen, leading to a finite value of $S_{e e}(0)$.

## C. Hydrogen: solid density

Let us next consider hydrogen at solid density, i.e., $r_{s}=3.23(\rho=0.08 \mathrm{~g} / \mathrm{cc})$ at $\Theta=1(T=4.80 \mathrm{eV})$. Such conditions can be realized e.g. in experiments with hydrogen jets [134] which can be optically heated and subsequently be probed with XRTS [135]. From a physical perspective, such conditions may give rise to interesting effects such as a non-monotonic dispersion relation of the dynamic structure factor at intermediate wave numbers [136, 137] resembling the roton feature known e.g. from ultracold helium [138-141]. From a technical perspective, lower densities are, generally, more challenging for theoretical methods due to the increased impact of XC-effects and the larger degree of inhomogeneity [142, 143]. In the case of PIMC, the same trend holds w.r.t the FSP as it has been explained during the discussion of Fig. 2 above.

In Fig. 7, we show extensive new PIMC results for the $\xi$-extrapolation of various structural properties at $\Theta=1$. Panel a) corresponds to $S_{e e}(q)$ and is flat to a high degree. This is mainly a consequence of the electronic localization around protons. At the same time, we find that
![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-10.jpg?height=1248&width=1588&top_left_y=172&top_left_x=255)

FIG. 8. Ab initio PIMC results for hydrogen with $N=14, r_{s}=3.23$, and $\Theta=1$. a) spin up-down PCF $g_{u d}(r)$, b) spin up-up PCF $\left.g_{u u}(r), \mathrm{c}\right)$ electron-proton PCF $\left.g_{e p}(r), \mathrm{d}\right)$ proton-proton PCF $g_{p p}(r)$. Green crosses: direct (exact) PIMC results for $\xi=-1$; red circles: $\xi$-extrapolated results [Eq. (3)]; grey area: FSP free domain of $\xi \in[0,1]$. The dotted blue line is panel c) has been computed from the ground-state wave function of an isolated hydrogen atom, and has been rescaled arbitrarily as a guide to the eye.

the $\xi$-extrapolation works with very high accuracy and reproduces the exact direct PIMC results for $\xi=-1$ for all $q$ within the given level of accuracy. The same holds for the thermal structure factor $F_{e e}(\mathbf{q}, \beta / 2)$, electron-proton $\mathrm{SSF} S_{e p}(\mathbf{q})$, and proton-proton SSF $S_{p p}(\mathbf{q})$ shown in panels b), c), and d), respectively.

In Fig. 8a), we show the spin-offdiagonal PCF $g_{u d}(r)$. First, we find a somewhat more pronounced impact of quantum statistics in the solid density case compared to $r_{s}=2$ that has been investigated in Fig. 3 above. Second, $g_{u d}(r)$ exhibits an interesting and nontrivial structure with a significant maximum around $r=0.5 \AA$. It is a sign of the formation of $\mathrm{H}^{-}$ions and the incipient formation of molecules in the system. Third, the $\xi$-extrapolation method again works well for all $r$. The corresponding spin-diagonal PCF $g_{u u}(r)$ is shown in Fig. 8b) and exhibits a strikingly different behaviour. In this case, the impact of quantum statistics is approximately $100 \%$, and we find a pronounced exchange-correlation hill around $r=0.5 \AA$ for $\xi=1$. Nevertheless, the $\xi$-extrapolation method accurately captures the correct XC-hole in the fermionic limit. We note that it holds $g_{u d}(r)=g_{u u}(r)$ for $\xi=0$. From a physical perspective, the bosonic effective attraction leads to a clustering of spin-aligned electrons and, in this way, to the formation of bosonic molecules. This can be seen particularly well in Fig. 8d), where we show the proton-proton PCF $g_{p p}(r)$. It exhibits a pronounced peak around the molecular distance of $r=0.74 \AA$ [53] that vanishes with decreasing $\xi$. It is entirely absent for $\xi=-1$, which is fully captured by the $\xi$-extrapolation method. Finally, panel Fig. 8c) shows the electron-proton PCF $g_{e p}(r)$. While it is qualitatively similar to the case of $r_{s}=2$ shown in Fig. 5 above, it exhibits a nearly flat progression for $a_{\mathrm{B}} \lesssim r \lesssim 2 a_{\mathrm{B}}$; this feature is indicative of a somewhat lower degree of electronic delocalization, as it is expected.
![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-11.jpg?height=1994&width=1610&top_left_y=252&top_left_x=255)

FIG. 9. Ab initio PIMC results for Be with $r_{s}=0.93(\rho=7.5 \mathrm{~g} / \mathrm{cc}), \Theta=1.73(T=100 \mathrm{eV})$, and $N=4$. a) electron-electron $\operatorname{SSF} S_{e e}(\mathbf{q})$, b) thermal electron-electron structure factor $F_{e e}(\mathbf{q}, \beta / 2)$, c) electron-proton SSF $S_{e p}(\mathbf{q})$, d) proton-proton SSF $S_{p p}(\mathbf{q})$, e) spin up-down PCF $g_{u d}(r)$, f) spin up-up PCF $g_{u u}(r)$. Green crosses: direct (exact) PIMC results for $\xi=-1$; red circles: $\xi$-extrapolated results [Eq. (3)]; grey area: FSP free domain of $\xi \in[0,1]$.

![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-12.jpg?height=561&width=764&top_left_y=191&top_left_x=214)

FIG. 10. The $\xi$-dependence of the electron-electron SSF $S_{\text {ee }}(\mathbf{q})$ for $N_{\text {atom }}=4, r_{s}=0.93$, and $\Theta=1.73$. Symbols: PIMC data for $q=3.14 \AA-1$ (black stars), $q=6.29 \AA-1$ (blue diamonds), and $q=14.06 \AA-1$ (green crosses). Solid red lines: fits according to Eq. (3) based on PIMC data in the FSP free domain of $\xi \in[0,1]$.

![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-12.jpg?height=1079&width=726&top_left_y=1079&top_left_x=236)

FIG. 11. PIMC results for the electronic SSF $S_{e e}(\mathbf{q})$ of Be $r_{s}=0.93$ and $\Theta=1.73$ for $N_{\text {atom }}=4$ (diamonds) and $N_{\text {atom }}=$ 10 (crosses) Be atoms. Top: impact of quantum statistics, with green, blue and red symbols showing results for $\xi=-1$ (extrapolated), $\xi=0$, and $\xi=1$; the yellow squares show direct PIMC results for $N_{\text {atom }}=4$ and $\xi=-1$. Bottom: comparing PIMC results for hydrogen to UEG results via ESA [121, 122].

## D. Strongly compressed beryllium

As a final example, we consider compressed Be at $r_{s}=0.93(\rho=7.5 \mathrm{~g} / \mathrm{cc})$ and $\Theta=1.73(T=100 \mathrm{eV})$. Such conditions have been realized in experiments at the NIF [96, 97], and have recently been studied with the $\xi$-extrapolation method in Ref. [95]. In Fig. 9, we show an analysis of the familiar set of structural properties computed for $N_{\text {atom }}=4$ Be atoms (i.e., $N=16$ electrons) with the usual color code. We observe the same qualitative trends as reported for hydrogen in the previous sections and restrict ourselves to a discussion of the key differences here. First and foremost, we find excellent agreement between the $\xi$-extrapolation and the exact direct PIMC simulation for $\xi=-1$ for all considered observables. This can be discerned more directly for the electron-electron SSF in Fig. 10, where we explicitly show the $\xi$-dependence for three selected wavenumbers $q$. Interestingly, $S_{e e}(\mathbf{q})$ exhibits a slightly though definite non-monotonic behaviour with a shallow minimum around intermediate $q$. It is a consequence of the interplay between the plasmon weight that decreases with $q$, and the increasing weight of the elastic feature in the long wavelength limit. In contrast, $F_{e e}(\mathbf{q}, \beta / 2), S_{e I}(\mathbf{q})$, and $S_{I I}(\mathbf{q})$ exhibit the same qualitative trends as reported previously for hydrogen. The spin-offdiagonal PCF $g_{u d}(r)$ shown in Fig. 9e) exhibits a substantial increase towards $r \rightarrow 0$; this is a direct consequence of the presence of ions with a fully occupied K-shell at these conditions, which is more pronounced in the bosonic case. This trend is completely absent for the spin-diagonal pendant $g_{u u}(r)$, where $g(r)=0$ holds by definition. At the same time, we point out the remarkable impact of quantum statistics for Be at these conditions with $g_{u u}^{\xi=1}(0) \approx 4$ for bosons. We also note that we find an average sign of $S \approx 0.11$ for $N_{\text {atom }}=4$ Be atoms in our direct PIMC calculations.

Finally, we study the dependence of the electronelectron SSF on the system size in Fig. 11. Specifically, the diamonds and crosses show our PIMC results for $N_{\text {atom }}=4$ and $N_{\text {atom }}=10$, and the red, blue and green symbols in the top panel correspond to $\xi=1, \xi=0$, and $\xi=-1$. Overall, we find the same qualitative trends as observed for hydrogen in Fig. 6 above, namely a substantial dependence on the number of particles in the bosonic case which is absent for boltzmannons and fermions. This conclusion is further substantiated by the yellow squares that show the exact, direct fermionic PIMC simulations for $N_{\text {atom }}=4$ and nicely agree with the $\xi$-extrapolated results for both system sizes.

In the bottom panel of Fig. 11, we compare our PIMC results for Be with the corresponding electronic SSF for the UEG at the same conditions; it has been computed within the aforementioned ESA [121, 122] and is depicted by the solid blue curve. Evidently, the Be system does not even qualitatively resemble the UEG model despite the comparably high temperature. More specifically, the localized K-shell electrons, as well as the loosely local-

![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-13.jpg?height=729&width=680&top_left_y=172&top_left_x=278)

FIG. 12. Electronic density distribution in a fixed ion snapshot for $N_{\text {atom }}=4$ Be atoms at $r_{s}=0.93$ and $\Theta=1.73$.

ized electronic screening cloud give rise to a pronounced elastic feature in the dynamic structure factor $S_{e e}(\mathbf{q}, \omega)$ in this regime $[61,95,96]$. This, in turn, leads to an increasing normalization $S_{e e}(\mathbf{q})$ [cf. Eq. (4)] for small $q$ despite the vanishing plasmon.

## E. Strongly compressed beryllium: snapshot

Let us conclude our investigation with a brief study of the $\xi$-extrapolation method applied to the electronic problem in a fixed external ion snapshot potential. A similar set-up has recently allowed Böhme et al. [103, 117] to present the first exact results for the electronic density response and the associated exchange-correlation kernel of warm dense hydrogen. In addition, such calculations provide direct information about the impact of the ions on the density response [144-146], and about the formation of bound states and molecules [53]. In Fig. 12, we show PIMC results for the electronic density computed for a snapshot of $N_{\text {atom }}=4$ Be atoms at the same conditions studied in Sec. III D. These results nicely illustrate a high degree of electronic localization around the nuclei, and a substantially reduced density in the interstitial region. From the perspective of the $\xi$-extrapolation method that constitutes the focus of the present work, one might expect that such a spatially resolved observable constitutes a most challenging benchmark case due to the comparably larger impact of quantum statistics effects in the vicinity of the nuclei.

In Fig. 13, we show the corresponding electronic density in the $y$-z-plane, and the $x$-positions have been chosen so that panels a) and b) approximately contain two ions, whereas panels c) and d) show the interstitial region without such nuclei. Further, the left column shows exact, direct fermionic PIMC results, whereas the right column has been computed from the $\xi$-extrapolation via Eq. (3) using as input PIMC results from the signproblem free domain of $\xi \in[0,1]$. Most importantly, we find no significant differences between the direct and the extrapolated results in both cases. This can be seen more clearly in Fig. 14, where we show scan lines over the $2 D$ densities. Let us first consider the left column that has been obtained for the layer that includes two ions, and the top and bottom panels correspond to the dotted and dashed lines in Fig. 13. As it is expected, we find a comparably larger localization in the bosonic limit around the nuclei. Moreover, the $\xi$-extrapolation works very well, although there appear small differences around the maximum of the density in the top panel. This, however, is not indicative of a statistically significant systematic underestimation of the true fermionic density in this region, and does not occur for the other nuclei (i.e., see the bottom panel). The right column shows the same analysis for the layer that is located in the interstitial region. In this case, the bosonic density is comparably reduced to the other data sets; this is required to balance the increased bosonic localization around the nuclei. Most importantly, the $\xi$-extrapolation fully captures the correct fermionic limit everywhere.

## IV. SUMMARY AND DISCUSSION

In this work, we have presented a detailed investigation of a variety of structural properties for warm dense hydrogen and beryllium. As the first test case, we have considered hydrogen at $r_{s}=2$ and $\Theta=1$, where most of the electrons are assumed to be free $[53,85,93]$. As a consequence, the system behaves qualitatively similar to the free UEG, where it is known that the $\xi$-extrapolation works well [77, 78]; the same does indeed hold for $\mathrm{H}$ at these conditions. The second test case concerns hydrogen at solid density, $r_{s}=3.23$ and $\Theta=1$. These conditions can be created in experiments with hydrogen jets $[134,135]$, and might give rise to a nontrivial rotontype feature at sufficiently high temperature [136, 137]. From a theoretical perspective, they constitute a more challenging example due to the comparably larger impact of quantum statistics that is reflected by the somewhat more severe fermion sign problem. In practice, we find that simulations based on Bose-Einstein statistics result in the formation of molecules, which are completely absent for a proper fermionic treatment of the electrons. At the same time, the $\xi$-extrapolation method nicely captures these qualitative differences and works equally well for all the investigated properties. The third physical system studied in this work concerns compressed Be at $r_{s}=0.93$ and $\Theta=1.73$; these conditions are relevant for experiments at the NIF $[96,97]$ and have very recently been studied with the $\xi$-extrapolation method by Dornheim et al. [95]. Unsurprisingly, the complex interplay between the more strongly charged Be nuclei and the
![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-14.jpg?height=1328&width=1740&top_left_y=184&top_left_x=152)

FIG. 13. Ab initio PIMC results for the electronic density in the $y$ - $z$-layer. Panels a) and b) have been computed for a layer with two ions in it, whereas panels c) and d) are taken from the interstitial region. The left and right columns show exact, direct fermionic PIMC calculations and corresponding $\xi$-extrapolated results based on input data from the FSP free region of $\xi \in[0,1]$, respectively. The dashed and dotted horizontal lines show scan lines that are investigated in more detail in Fig. 14.

electrons gives rise to a richer physics that is reflected by an even more pronounced impact of quantum statistics compared to hydrogen despite the larger value of $\Theta$. This includes the partial double-occupation of the K-shell, which strongly depends on the spin-polarization of the involved electrons, giving rise to substantial differences in the spin-resolved pair correlation functions. Nevertheless, the $\xi$-extrapolation method works exceptionally well despite these complexities. Finally, we have used PIMC to solve the electronic problem in the external potential of a fixed configuration of $N_{\text {atom }}=4$ Be nuclei. From a technical perspective, analyzing the corresponding electronic density distribution might constitute the most challenging benchmark case that we have considered in this work as it allows us to spatially resolve the impact of quantum statistics, whereas potential systematic errors might be averaged out in aggregated properties such as the SSFs and PCFs considered before. In practice, the $\xi$-extrapolation method works remarkably well both in the vicinity of the nuclei, as well as in the interstitial region. All PIMC results that have been presented here are freely available online [113] and can be used to benchmark new methods and approximations.

We are convinced that these findings open up a variety of potential projects for future works. While the $\xi$-extrapolation method has worked exceptionally well in all presently studied cases, previous findings for the warm dense UEG model [77, 78] indicate that this idea breaks down for $\Theta<1$. The particular limits will likely strongly depend both on the density and the elemental composition. A detailed study of the applicability range of the method for various light elements and their mixtures thus constitutes an important task for dedicated future works.

A particular strength of the $\xi$-extrapolation method is given by its polynomial scaling with the system size, which combines a number of advantages. First, this ac-
![](https://cdn.mathpix.com/cropped/2024_06_04_20ceaa868a059243627fg-15.jpg?height=1336&width=1594&top_left_y=172&top_left_x=255)

FIG. 14. Scan lines of the electronic density for the dashed (top) and dotted (bottom) horizontal lines in Fig. 13. The left and right columns correspond to the layer with and without ions in it.

cess to comparably large systems allows one to study finite-size effects, which are small for the structural properties studied here, but may play a more important role for the computation of an equation-of-state $[85,93]$. The latter are of direct importance for a number of applications including laser fusion [10], and are thus of considerable interest in their own right. Second, simulating large systems allows one to compute properties such as the ITCF in the limit of small $q$. This is important e.g. to describe XRTS experiments in a forward-scattering geometry [95], and might be needed to fully capture dynamic long-range effects in other properties [147].

An additional direction for future research is the dedicated study of density response properties. In this regard, we note the excellent performance of the $\xi$-extrapolation method w.r.t. the ITCF $F_{e e}(\mathbf{q}, \tau)$, which can be used as input for the imaginary-time version of the fluctuationdissipation theorem $[60,148]$

$$
\begin{equation*}
\chi_{a b}(\mathbf{q}, 0)=-\frac{\sqrt{N_{a} N_{b}}}{\Omega} \int_{0}^{\beta} \mathrm{d} \tau F_{a b}(\mathbf{q}, \tau) \tag{5}
\end{equation*}
$$

In this way, one can get direct access to the speciesresolved static density response function $\chi_{a b}(\mathbf{q}, 0)$ and, in this way, a set of exchange-correlation properties such as the electron-ion local field correction of different systems. Moreover, the corresponding estimation of higherorder imaginary-time correlation functions can be used as input for similar relations [66] that give one access to a variety of nonlinear response properties [63-65, 149].

Finally, we note that the impact of quantum statistics is comparably large for Coulomb interacting systems such as in warm dense matter, but may be substantially smaller for more short-range repulsive interactions [119]. This opens up the intriguing possibility to study ultracold fermionic atoms such as ${ }^{3} \mathrm{He}[138,140,150]$ with unprecedented accuracy. Given the above, returning to two-component systems, the $\xi$-extrapolation method can also improve our rather poor understanding of liquid ${ }^{3} \mathrm{He}-{ }^{4} \mathrm{He}$ mixtures at low temperatures. Binary bosonfermion mixtures have traditionally gathered strong interest owing to the possibility of double superfluidity and
the unique correlation-driven interplay between different quantum statistics [151-153]. Nevertheless, ab initio simulations of such systems are essentially non-existent. To our knowledge, finite temperature isotopic helium mixtures have been so far investigated within the fixed-node approximation [154-157] or with the PIMC method but neglecting fermionic exchange effects [158]. The quantities of interest have been limited to the superfluid fraction, the component kinetic energies, the momentum distribution functions and the pair correlation functions. Thus, the $\xi$-extrapolation technique, being sign problem free inside the phase diagram region of its applicability, can be employed for the acquisition of extensive quasiexact thermodynamic and structural results as well as for the first reconstruction of collective excitation spectra.

## ACKNOWLEDGMENTS

This work was partially supported by the Center for Advanced Systems Understanding (CASUS), financed by Germany's Federal Ministry of Education and Research (BMBF) and the Saxon state government out of the State budget approved by the Saxon State Parliament. This work has received funding from the European Research Council (ERC) under the European Union's Horizon 2022 research and innovation programme (Grant agreement No. 101076233, "PREXTREME"). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. Computations were performed on a Bull Cluster at the Center for Information Services and High-Performance Computing (ZIH) at Technische Universität Dresden, at the Norddeutscher Verbund für Hoch- und Höchstleistungsrechnen (HLRN) under grant mvp00024, and on the HoreKa supercomputer funded by the Ministry of Science, Research and the Arts BadenWürttemberg and by the Federal Ministry of Education and Research.
[1] F. Graziani, M. P. Desjarlais, R. Redmer, and S. B. Trickey, eds., Frontiers and Challenges in Warm Dense Matter (Springer, International Publishing, 2014).

[2] Alessandra Benuzzi-Mounaix, Stéphane Mazevet, Alessandra Ravasio, Tommaso Vinci, Adrien Denoeud, Michel Koenig, Nourou Amadou, Erik Brambrink, Floriane Festa, Anna Levy, Marion Harmand, Stéphanie Brygoo, Gael Huser, Vanina Recoules, Johan Bouchet, Guillaume Morard, François Guyot, Thibaut de Resseguier, Kohei Myanishi, Norimasa Ozaki, Fabien Dorchies, Jerôme Gaudin, Pierre Marie Leguay, Olivier Peyrusse, Olivier Henry, Didier Raffestin, Sebastien Le Pape, Ray Smith, and Riccardo Musella, "Progress in warm dense matter study with applications to planetology," Phys. Scripta T161, 014060 (2014).

[3] A. Becker, W. Lorenzen, J. J. Fortney, N. Nettelmann, M. Schöttler, and R. Redmer, "Ab initio equations of state for hydrogen (h-reos.3) and helium (he-reos.3) and their implications for the interior of brown dwarfs," Astrophys. J. Suppl. Ser 215, 21 (2014).

[4] P. Haensel, A. Y. Potekhin, and D. G. Yakovlev, eds., "Equilibrium plasma properties. outer envelopes," in Neutron Stars 1 (Springer New York, New York, NY, 2007) pp. 53-114.

[5] Mark L. Brongersma, Naomi J. Halas, and Peter Nordlander, "Plasmon-induced hot carrier science and technology," Nature Nanotechnology 10, 25-34 (2015).

[6] A. Lazicki, D. McGonegle, J. R. Rygg, D. G. Braun, D. C. Swift, M. G. Gorman, R. F. Smith, P. G. Heighway, A. Higginbotham, M. J. Suggit, D. E. Fratanduono, F. Coppari, C. E. Wehrenberg, R. G. Kraus, D. Erskine, J. V. Bernier, J. M. McNaney, R. E. Rudd,
G. W. Collins, J. H. Eggert, and J. S. Wark, "Metastability of diamond ramp-compressed to 2 terapascals," Nature 589, 532-535 (2021).

[7] D. Kraus, A. Ravasio, M. Gauthier, D. O. Gericke, J. Vorberger, S. Frydrych, J. Helfrich, L. B. Fletcher, G. Schaumann, B. Nagler, B. Barbrel, B. Bachmann, E. J. Gamboa, S. Göde, E. Granados, G. Gregori, H. J. Lee, P. Neumayer, W. Schumaker, T. Döppner, R. W. Falcone, S. H. Glenzer, and M. Roth, "Nanosecond formation of diamond and lonsdaleite by shock compression of graphite," Nature Communications 7, 10970 (2016).

[8] D. Kraus, J. Vorberger, A. Pak, N. J. Hartley, L. B. Fletcher, S. Frydrych, E. Galtier, E. J. Gamboa, D. O. Gericke, S. H. Glenzer, E. Granados, M. J. MacDonald, A. J. MacKinnon, E. E. McBride, I. Nam, P. Neumayer, M. Roth, A. M. Saunders, A. K. Schuster, P. Sun, T. van Driel, T. Döppner, and R. W. Falcone, "Formation of diamonds in laser-compressed hydrocarbons at planetary interior conditions," Nature Astronomy 1, 606-611 (2017).

[9] R. Betti and O. A. Hurricane, "Inertial-confinement fusion with lasers," Nature Physics 12, 435-448 (2016).

[10] S. X. Hu, B. Militzer, V. N. Goncharov, and S. Skupsky, "First-principles equation-of-state table of deuterium for inertial confinement fusion applications," Phys. Rev. B 84, 224109 (2011).

[11] Abu-Shawareb et al. (The Indirect Drive ICF Collaboration), "Achievement of target gain larger than unity in an inertial fusion experiment," Phys. Rev. Lett. 132, 065102 (2024).

[12] Dimitri Batani, Arnaud Colaïtis, Fabrizio Consoli, Colin N. Danson, Leonida Antonio Gizzi, Javier Honrubia, Thomas Kühl, Sebastien Le Pape, Jean-Luc Miquel, Jose Manuel Perlado, and et al., "Future for inertial-fusion energy in europe: a roadmap," High Power Laser Science and Engineering 11, e83 (2023).

[13] T. Dornheim, S. Groth, and M. Bonitz, "The uniform electron gas at warm dense matter conditions," Phys. Reports 744, 1-86 (2018).

[14] Tobias Dornheim, Zhandos A. Moldabekov, Kushal Ramakrishna, Panagiotis Tolias, Andrew D. Baczewski, Dominik Kraus, Thomas R. Preston, David A. Chapman, Maximilian P. Böhme, Tilo Döppner, Frank Graziani, Michael Bonitz, Attila Cangi, and Jan Vorberger, "Electronic density response of warm dense matter," Physics of Plasmas 30, 032705 (2023).

[15] M. Bonitz, T. Dornheim, Zh. A. Moldabekov, S. Zhang, P. Hamann, H. Kählert, A. Filinov, K. Ramakrishna, and J. Vorberger, "Ab initio simulation of warm dense matter," Physics of Plasmas 27, 042710 (2020).

[16] G. Giuliani and G. Vignale, Quantum Theory of the Electron Liquid (Cambridge University Press, Cambridge, 2008).

[17] Torben Ott, Hauke Thomsen, Jan Willem Abraham, Tobias Dornheim, and Michael Bonitz, "Recent progress in the theory and simulation of strongly correlated plasmas: phase transitions, transport, quantum, and magnetic field effects," The European Physical Journal D 72, 84 (2018).

[18] N. David Mermin, "Thermal properties of the inhomogeneous electron gas," Phys. Rev. 137, A1441-A1443 (1965).

[19] D. M. Ceperley and B. J. Alder, "Ground state of the electron gas by a stochastic method," Phys. Rev. Lett. 45, 566-569 (1980).

[20] G. G. Spink, R. J. Needs, and N. D. Drummond, "Quantum Monte Carlo study of the three-dimensional spin-polarized homogeneous electron gas," Phys. Rev. B 88, 085121 (2013).

[21] S. Moroni, D. M. Ceperley, and G. Senatore, "Static response from quantum Monte Carlo calculations," Phys. Rev. Lett 69, 1837 (1992).

[22] S. Moroni, D. M. Ceperley, and G. Senatore, "Static response and local field factor of the electron gas," Phys. Rev. Lett 75, 689 (1995).

[23] R. O. Jones, "Density functional theory: Its origins, rise to prominence, and future," Rev. Mod. Phys. 87, 897923 (2015).

[24] V. V. Karasiev, L. Calderin, and S. B. Trickey, "Importance of finite-temperature exchange correlation for warm dense matter calculations," Phys. Rev. E 93, 063207 (2016).

[25] M. W. C. Dharma-wardana, "Current issues in finite-t density-functional theory and warm-correlated matter," Computation 4 (2016), 10.3390/computation4020016.

[26] Zhandos A. Moldabekov, Mani Lokamani, Jan Vorberger, Attila Cangi, and Tobias Dornheim, "Nonempirical Mixing Coefficient for Hybrid XC Functionals from Analysis of the XC Kernel," The Journal of Physical Chemistry Letters 14, 1326-1333 (2023).

[27] D. M. Ceperley, "Path integrals in the theory of condensed helium," Rev. Mod. Phys 67, 279 (1995).

[28] Minoru Takahashi and Masatoshi Imada, "Monte Carlo Calculation of Quantum Systems," Journal of the Phys- ical Society of Japan 53, 963-974 (1984).

[29] M. F. Herman, E. J. Bruskin, and B. J. Berne, "On path integral Monte Carlo simulations," The Journal of Chemical Physics 76, 5150-5155 (1982).

[30] E. L. Pollock and D. M. Ceperley, "Path-integral computation of superfluid densities," Phys. Rev. B 36, $8343-8352$ (1987).

[31] T. Dornheim, A. Filinov, and M. Bonitz, "Superfluidity of strongly correlated bosons in two- and threedimensional traps," Phys. Rev. B 91, 054503 (2015).

[32] Kwangsik Nho and D. P. Landau, "Bose-Einstein condensation temperature of a homogeneous weakly interacting Bose gas: Path integral Monte Carlo study," Phys. Rev. A 70, 053614 (2004).

[33] Hiroki Saito, "Path-Integral Monte Carlo Study on a Droplet of a Dipolar Bose-Einstein Condensate Stabilized by Quantum Fluctuation," Journal of the Physical Society of Japan 85, 053001 (2016).

[34] T. Dornheim, "Fermion sign problem in path integral Monte Carlo simulations: Quantum dots, ultracold atoms, and warm dense matter," Phys. Rev. E 100, 023307 (2019).

[35] Tobias Dornheim, "Fermion sign problem in path integral monte carlo simulations: grand-canonical ensemble," Journal of Physics A: Mathematical and Theoretical 54, 335001 (2021).

[36] M. Troyer and U. J. Wiese, "Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations," Phys. Rev. Lett 94, 170201 (2005).

[37] Ethan W. Brown, Bryan K. Clark, Jonathan L. DuBois, and David M. Ceperley, "Path-Integral Monte Carlo Simulation of the Warm Dense Homogeneous Electron Gas," Phys. Rev. Lett. 110, 146405 (2013).

[38] T. Schoof, S. Groth, J. Vorberger, and M. Bonitz, "Ab initio thermodynamic results for the degenerate electron gas at finite temperature," Phys. Rev. Lett. 115, 130402 (2015).

[39] Tobias Dornheim, Simon Groth, Alexey Filinov, and Michael Bonitz, "Permutation blocking path integral Monte Carlo: a highly efficient approach to the simulation of strongly degenerate non-ideal fermions," New Journal of Physics 17, 073017 (2015).

[40] T. Dornheim, S. Groth, T. Sjostrom, F. D. Malone, W. M. C. Foulkes, and M. Bonitz, "Ab initio quantum Monte Carlo simulation of the warm dense electron gas in the thermodynamic limit," Phys. Rev. Lett. 117, 156403 (2016).

[41] Fionn D. Malone, N. S. Blunt, James J. Shepherd, D. K. K. Lee, J. S. Spencer, and W. M. C. Foulkes, "Interaction picture density matrix quantum Monte Carlo," The Journal of Chemical Physics 143, 044116 (2015).

[42] Fionn D. Malone, N. S. Blunt, Ethan W. Brown, D. K. K. Lee, J. S. Spencer, W. M. C. Foulkes, and James J. Shepherd, "Accurate exchange-correlation energies for the warm dense electron gas," Phys. Rev. Lett. 117,115701 (2016).

[43] Joonho Lee, Miguel A. Morales, and Fionn D. Malone, "A phaseless auxiliary-field quantum Monte Carlo perspective on the uniform electron gas at finite temperatures: Issues, observations, and benchmark study," The Journal of Chemical Physics 154, 064109 (2021).

[44] A. Yilmaz, K. Hunger, T. Dornheim, S. Groth, and M. Bonitz, "Restricted configuration path integral Monte Carlo," The Journal of Chemical Physics 153, 124114 (2020).

[45] Tobias Dornheim, Simon Groth, and Michael Bonitz, "Permutation blocking path integral monte carlo simulations of degenerate electrons at finite temperature," Contributions to Plasma Physics 59, e201800157 (2019).

[46] Tobias Dornheim, Simon Groth, Fionn D. Malone, Tim Schoof, Travis Sjostrom, W. M. C. Foulkes, and Michael Bonitz, "Ab initio quantum Monte Carlo simulation of the warm dense electron gas," Physics of Plasmas 24, 056303 (2017).

[47] Valentin V. Karasiev, Travis Sjostrom, James Dufty, and S. B. Trickey, "Accurate homogeneous electron gas exchange-correlation free energy for local spin-density calculations," Phys. Rev. Lett. 112, 076403 (2014).

[48] S. Groth, T. Dornheim, T. Sjostrom, F. D. Malone, W. M. C. Foulkes, and M. Bonitz, "Ab initio exchangecorrelation free energy of the uniform electron gas at warm dense matter conditions," Phys. Rev. Lett. 119, 135001 (2017).

[49] Valentin V. Karasiev, S. B. Trickey, and James W. Dufty, "Status of free-energy representations for the homogeneous electron gas," Phys. Rev. B 99, 195134 (2019).

[50] Gerd Röpke, Tobias Dornheim, Jan Vorberger, David Blaschke, and Biplab Mahato, "Virial coefficients of the uniform electron gas from path integral Monte Carlo simulations," Phys. Rev. E 109, 025202 (2024).

[51] Travis Sjostrom and Jérôme Daligault, "Gradient corrections to the exchange-correlation free energy," Phys. Rev. B 90, 155109 (2014)

[52] Kushal Ramakrishna, Tobias Dornheim, and Jan Vorberger, "Influence of finite temperature exchangecorrelation effects in hydrogen," Phys. Rev. B 101, 195129 (2020).

[53] Zhandos Moldabekov, Sebastian Schwalbe, Maximilian P. Böhme, Jan Vorberger, Xuecheng Shao, Michele Pavanello, Frank R. Graziani, and Tobias Dornheim, "Bound-state breaking and the importance of thermal exchange-correlation effects in warm dense hydrogen," Journal of Chemical Theory and Computation 20, 6878 (2024).

[54] Zhandos Moldabekov, Maximilian Böhme, Jan Vorberger, David Blaschke, and Tobias Dornheim, "Ab initio static exchange-correlation kernel across jacob's ladder without functional derivatives," Journal of Chemical Theory and Computation 19, 1286-1299 (2023).

[55] Zhandos Moldabekov, Tobias Dornheim, Maximilian Böhme, Jan Vorberger, and Attila Cangi, "The relevance of electronic perturbations in the warm dense electron gas," The Journal of Chemical Physics 155, 124116 (2021).

[56] Zhandos Moldabekov, Tobias Dornheim, Jan Vorberger, and Attila Cangi, "Benchmarking exchange-correlation functionals in the spin-polarized inhomogeneous electron gas under warm dense conditions," Phys. Rev. B 105, 035134 (2022).

[57] A. Pribram-Jones, P. E. Grabowski, and K. Burke, "Thermal density functional theory: Time-dependent linear response and approximate functionals from the fluctuation-dissipation theorem," Phys. Rev. Lett 116,
233001 (2016).

[58] Tobias Dornheim, Maximilian Böhme, Dominik Kraus, Tilo Döppner, Thomas R. Preston, Zhandos A. Moldabekov, and Jan Vorberger, "Accurate temperature diagnostics for matter under extreme conditions," Nature Communications 13, 7911 (2022).

[59] Tobias Dornheim, Maximilian P. Böhme, David A. Chapman, Dominik Kraus, Thomas R. Preston, Zhandos A. Moldabekov, Niclas Schlünzen, Attila Cangi, Tilo Döppner, and Jan Vorberger, "Imaginary-time correlation function thermometry: A new, high-accuracy and model-free temperature analysis technique for $\mathrm{x}$ ray Thomson scattering data," Physics of Plasmas 30, 042707 (2023).

[60] Tobias Dornheim, Zhandos Moldabekov, Panagiotis Tolias, Maximilian Böhme, and Jan Vorberger, "Physical insights from imaginary-time density-density correlation functions," Matter and Radiation at Extremes 8, 056601 (2023).

[61] Maximilian P. Böhme, Luke B. Fletcher, Tilo Döppner, Dominik Kraus, Andrew D. Baczewski, Thomas R. Preston, Michael J. MacDonald, Frank R. Graziani, Zhandos A. Moldabekov, Jan Vorberger, and Tobias Dornheim, "Evidence of free-bound transitions in warm dense matter and their impact on equation-of-state measurements," (2023), arXiv:2306.17653 [physics.plasm$\mathrm{ph}]$.

[62] Tobias Dornheim, Tilo Döppner, Andrew D. Baczewski, Panagiotis Tolias, Maximilian P. Böhme, Zhandos A. Moldabekov, Divyanshu Ranjan, David A. Chapman, Michael J. MacDonald, Thomas R. Preston, Dominik Kraus, and Jan Vorberger, "X-ray Thomson scattering absolute intensity from the f-sum rule in the imaginary-time domain," arXiv (2023), 2305.15305 [physics.plasm-ph].

[63] Tobias Dornheim, Jan Vorberger, and Zhandos A. Moldabekov, "Nonlinear density response and higher order correlation functions in warm dense matter," Journal of the Physical Society of Japan 90, 104002 (2021).

[64] Tobias Dornheim, Jan Vorberger, and Michael Bonitz, "Nonlinear electronic density response in warm dense matter," Phys. Rev. Lett. 125, 085001 (2020).

[65] Tobias Dornheim, Maximilian Böhme, Zhandos A. Moldabekov, Jan Vorberger, and Michael Bonitz, "Density response of the warm dense electron gas beyond linear response theory: Excitation of harmonics," Phys. Rev. Research 3, 033231 (2021).

[66] Tobias Dornheim, Zhandos A. Moldabekov, and Jan Vorberger, "Nonlinear density response from imaginarytime correlation functions: Ab initio path integral Monte Carlo simulations of the warm dense electron gas," The Journal of Chemical Physics 155, 054110 (2021).

[67] Panagiotis Tolias, Tobias Dornheim, Zhandos A. Moldabekov, and Jan Vorberger, "Unravelling the nonlinear ideal density response of many-body systems," EPL 142,44001 (2023).

[68] Siu A. Chin, "High-order path-integral Monte Carlo methods for solving quantum dot problems," Phys. Rev. E 91, 031301 (2015).

[69] Tobias Dornheim, Tim Schoof, Simon Groth, Alexey Filinov, and Michael Bonitz, "Permutation blocking path integral monte carlo approach to the uniform electron gas at finite temperature," The Journal of Chemical Physics 143, 204101 (2015).

[70] Barak Hirshberg, Michele Invernizzi, and Michele Parrinello, "Path integral molecular dynamics for fermions: Alleviating the sign problem with the Bogoliubov inequality," The Journal of Chemical Physics 152, 171102 (2020).

[71] Tobias Dornheim, Michele Invernizzi, Jan Vorberger, and Barak Hirshberg, "Attenuating the fermion sign problem in path integral Monte Carlo simulations using the Bogoliubov inequality and thermodynamic integration," The Journal of Chemical Physics 153, 234104 (2020).

[72] D. M. Ceperley, "Fermion nodes," Journal of Statistical Physics 63, 1237-1267 (1991).

[73] K. P. Driver and B. Militzer, "All-electron path integral monte carlo simulations of warm dense matter: Application to water and carbon plasmas," Phys. Rev. Lett. 108,115502 (2012).

[74] Burkhard Militzer and Kevin P. Driver, "Development of path integral monte carlo simulations with localized nodal surfaces for second-row elements," Phys. Rev. Lett. 115, 176403 (2015).

[75] Yunuo Xiong and Hongwei Xiong, "On the thermodynamic properties of fictitious identical particles and the application to fermion sign problem," The Journal of Chemical Physics 157, 094112 (2022).

[76] Yunuo Xiong and Hongwei Xiong, "Thermodynamics of fermions at any temperature based on parametrized partition function," Phys. Rev. E 107, 055308 (2023).

[77] Tobias Dornheim, Panagiotis Tolias, Simon Groth, Zhandos A. Moldabekov, Jan Vorberger, and Barak Hirshberg, "Fermionic physics from ab initio path integral Monte Carlo simulations of fictitious identical particles," The Journal of Chemical Physics 159, 164113 (2023).

[78] Tobias Dornheim, Sebastian Schwalbe, Zhandos A. Moldabekov, Jan Vorberger, and Panagiotis Tolias, "Ab initio path integral Monte Carlo simulations of the uniform electron gas on large length scales," J. Phys. Chem. Lett. 15, 1305-1313 (2024).

[79] T. Schoof, M. Bonitz, A. Filinov, D. Hochstuhl, and J.W. Dufty, "Configuration Path Integral Monte Carlo," Contributions to Plasma Physics 51, 687-697 (2011).

[80] T. Schoof, S. Groth, and M. Bonitz, "Towards ab initio thermodynamics of the electron gas at strong degeneracy," Contributions to Plasma Physics 55, 136-143 (2015).

[81] S. Groth, T. Schoof, T. Dornheim, and M. Bonitz, "Ab initio quantum Monte Carlo simulations of the uniform electron gas without fixed nodes," Phys. Rev. B 93, 085102 (2016).

[82] N. S. Blunt, T. W. Rogers, J. S. Spencer, and W. M. C. Foulkes, "Density-matrix quantum Monte Carlo method," Phys. Rev. B 89, 245124 (2014).

[83] Tobias Dornheim, Maximilian Böhme, Burkhard Militzer, and Jan Vorberger, "Ab initio path integral monte carlo approach to the momentum distribution of the uniform electron gas at finite temperature without fixed nodes," Phys. Rev. B 103, 205142 (2021).

[84] B. Militzer, W. B. Hubbard, J. Vorberger, I. Tamblyn, and S. A. Bonev, "A massive core in jupiter predicted from first-principles simulations," The Astrophysical Journal 688, L45-L48 (2008).

[85] B. Militzer and D. M. Ceperley, "Path integral Monte Carlo simulation of the low-density hydrogen plasma,"
Phys. Rev. E 63, 066404 (2001).

[86] Burkhard Militzer, Felipe González-Cataldo, Shuai Zhang, Kevin P. Driver, and Fran çois Soubiran, "Firstprinciples equation of state database for warm dense matter computation," Phys. Rev. E 103, 013203 (2021).

[87] V. S. Filinov, V. E. Fortov, M. Bonitz, and Zh. Moldabekov, "Fermionic path-integral Monte Carlo results for the uniform electron gas at finite temperature," Phys. Rev. E 91, 033108 (2015).

[88] V. S. Filinov, M. Bonitz, V. E. Fortov, W. Ebeling, P. Levashov, and M. Schlanges, "Thermodynamic properties and plasma phase transition in dense hydrogen," Contributions to Plasma Physics 44, 388-394 (2004).

[89] V.S. Filinov, Yu.B. Ivanov, M. Bonitz, V.E. Fortov, and P.R. Levashov, "Color path-integral monte carlo simulations of quark-gluon plasma," Physics Letters A 376, 1096-1101 (2012).

[90] K. Sakkos, J. Casulleras, and J. Boronat, "High order Chin actions in path integral Monte Carlo," The Journal of Chemical Physics 130, 204109 (2009).

[91] T. Dornheim, S. Groth, T. Schoof, C. Hann, and M. Bonitz, "Ab initio quantum Monte Carlo simulations of the uniform electron gas without fixed nodes: The unpolarized case," Phys. Rev. B 93, 205134 (2016).

[92] T. Dornheim, S. Groth, J. Vorberger, and M. Bonitz, "Permutation blocking path integral Monte Carlo approach to the static density response of the warm dense electron gas," Phys. Rev. E 96, 023203 (2017).

[93] A. V. Filinov and M. Bonitz, "Equation of state of partially ionized hydrogen and deuterium plasma revisited," Phys. Rev. E 108, 055212 (2023).

[94] E. Prodan and W. Kohn, "Nearsightedness of electronic matter," Proceedings of the National Academy of Sciences 102, 11635-11638 (2005).

[95] Tobias Dornheim, Tilo Döppner, Panagiotis Tolias, Maximilian Böhme, Luke Fletcher, Thomas Gawne, Frank Graziani, Dominik Kraus, Michael MacDonald, Zhandos Moldabekov, Sebastian Schwalbe, Dirk Gericke, and Jan Vorberger, "Unraveling electronic correlations in warm dense quantum plasmas," (2024), arXiv:2402.19113 [physics.plasm-ph].

[96] T. Döppner, M. Bethkenhagen, D. Kraus, P. Neumayer, D. A. Chapman, B. Bachmann, R. A. Baggott, M. P. Böhme, L. Divol, R. W. Falcone, L. B. Fletcher, O. L. Landen, M. J. MacDonald, A. M. Saunders, M. Schörner, P. A. Sterne, J. Vorberger, B. B. L. Witte, A. Yi, R. Redmer, S. H. Glenzer, and D. O Gericke, "Observing the onset of pressure-driven k-shell delocalization," Nature 618, 270-275 (2023).

[97] M. J. MacDonald, C. A. Di Stefano, T. Döppner, L. B. Fletcher, K. A. Flippo, D. Kalantar, E. C. Merritt, S. J. Ali, P. M. Celliers, R. Heredia, S. Vonhof, G. W. Collins, J. A. Gaffney, D. O. Gericke, S. H. Glenzer, D. Kraus, A. M. Saunders, D. W. Schmidt, C. T. Wilson, R. Zacharias, and R. W. Falcone, "The colliding planar shocks platform to study warm dense matter at the National Ignition Facility," Physics of Plasmas 30, 062701 (2023).

[98] Louisa M. Fraser, W. M. C. Foulkes, G. Rajagopal, R. J. Needs, S. D. Kenny, and A. J. Williamson, "Finitesize effects and coulomb interactions in quantum monte carlo calculations for homogeneous systems with periodic boundary conditions," Phys. Rev. B 53, 1814-1832 (1996).

[99] M. Boninsegni, N. V. Prokofev, and B. V. Svistunov, "Worm algorithm and diagrammatic Monte Carlo: A new approach to continuous-space path integral Monte Carlo simulations," Phys. Rev. E 74, 036701 (2006).

[100] Hans De Raedt and Bart De Raedt, "Applications of the generalized Trotter formula," Phys. Rev. A 28, 35753580 (1983)

[101] B. Militzer, "Computation of the high temperature coulomb density matrix in periodic boundary conditions," Computer Physics Communications 204, 88-96 (2016).

[102] E.L. Pollock, "Properties and computation of the coulomb pair density matrix," Computer Physics Communications 52, 49-60 (1988).

[103] Maximilian Böhme, Zhandos A. Moldabekov, Jan Vorberger, and Tobias Dornheim, "Ab initio path integral monte carlo simulations of hydrogen snapshots at warm dense matter conditions," Phys. Rev. E 107, 015206 (2023).

[104] David Chandler and Peter G. Wolynes, "Exploiting the isomorphism between quantum theory and classical statistical mechanics of polyatomic fluids," The Journal of Chemical Physics 74, 4078-4095 (1981).

[105] Eran Rabani, David R. Reichman, Goran Krilov, and Bruce J. Berne, "The calculation of transport properties in quantum liquids using the maximum entropy numerical analytic continuation method: Application to liquid para-hydrogen," Proceedings of the National Academy of Sciences 99, 1129-1133 (2002).

[106] A. Filinov and M. Bonitz, "Collective and single-particle excitations in two-dimensional dipolar bose gases," Phys. Rev. A 86, 043628 (2012).

[107] T. Dornheim, S. Groth, J. Vorberger, and M. Bonitz, "Ab initio path integral Monte Carlo results for the dynamic structure factor of correlated electrons: From the electron liquid to warm dense matter," Phys. Rev. Lett. 121, 255001 (2018).

[108] S. Groth, T. Dornheim, and J. Vorberger, "Ab initio path integral Monte Carlo approach to the static and dynamic density response of the uniform electron gas," Phys. Rev. B 99, 235122 (2019).

[109] Paul Hamann, Tobias Dornheim, Jan Vorberger, Zhandos A. Moldabekov, and Michael Bonitz, "Dynamic properties of the warm dense electron gas based on abinitio path integral monte carlo simulations," Phys. Rev. B 102, 125150 (2020).

[110] Nicholas Metropolis, Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller, "Equation of state calculations by fast computing machines," The Journal of Chemical Physics 21, 1087-1092 (1953).

[111] M. Boninsegni, N. V. Prokofev, and B. V. Svistunov, "Worm algorithm for continuous-space path integral Monte Carlo simulations," Phys. Rev. Lett 96, 070601 (2006).

[112] Tobias Dornheim, Maximilian Böhme, and Sebastian Schwalbe, "ISHTAR - Imaginary-time Stochastic Highperformance Tool for Ab initio Research," (2024).

[113] A link to a repository containing all PIMC raw data will be made available upon publication.

[114] Tobias Dornheim, Panagiotis Tolias, Zhandos A. Moldabekov, Attila Cangi, and Jan Vorberger, "Effective electronic forces and potentials from ab initio path integral Monte Carlo simulations," The Journal of Chemical
Physics 156, 244113 (2022).

[115] T. Dornheim, S. Groth, A. V. Filinov, and M. Bonitz, "Path integral Monte Carlo simulation of degenerate electrons: Permutation-cycle properties," The Journal of Chemical Physics 151, 014108 (2019).

[116] E. I. Moses, R. N. Boyd, B. A. Remington, C. J. Keane, and R. Al-Ayat, "The national ignition facility: Ushering in a new age for high energy density science," Physics of Plasmas 16, 041006 (2009).

[117] Maximilian Böhme, Zhandos A. Moldabekov, Jan Vorberger, and Tobias Dornheim, "Static electronic density response of warm dense hydrogen: Ab initio path integral monte carlo simulations," Phys. Rev. Lett. 129, 066402 (2022).

[118] Tobias Dornheim, "Path-integral monte carlo simulations of quantum dipole systems in traps: Superfluidity, quantum statistics, and structural properties," Phys. Rev. A 102, 023307 (2020).

[119] Tobias Dornheim and Yangqian Yan, "Abnormal quantum moment of inertia and structural properties of electrons in 2D and 3D quantum dots: an ab initio pathintegral Monte Carlo study," New Journal of Physics 24, 113024 (2022).

[120] Tobias Dornheim, Jan Vorberger, Zhandos A. Moldabekov, and Panagiotis Tolias, "Spin-resolved density response of the warm dense electron gas," Phys. Rev. Research 4, 033018 (2022).

[121] Tobias Dornheim, Attila Cangi, Kushal Ramakrishna, Maximilian Böhme, Shigenori Tanaka, and Jan Vorberger, "Effective static approximation: A fast and reliable tool for warm-dense matter theory," Phys. Rev. Lett. 125, 235001 (2020).

[122] Tobias Dornheim, Zhandos A. Moldabekov, and Panagiotis Tolias, "Analytical representation of the local field correction of the uniform electron gas within the effective static approximation," Phys. Rev. B 103, 165102 (2021).

[123] Kai Hunger, Tim Schoof, Tobias Dornheim, Michael Bonitz, and Alexey Filinov, "Momentum distribution function and short-range correlations of the warm dense electron gas: Ab initio quantum monte carlo results," Phys. Rev. E 103, 053204 (2021).

[124] Panagiotis Tolias, Federico Lucco Castello, Fotios Kalkavouras, and Tobias Dornheim, "Revisiting the Vashishta-Singwi dielectric scheme for the warm dense uniform electron fluid," (2024), arXiv:2401.08502 [cond-mat.quant-gas].

[125] Tobias Dornheim and Jan Vorberger, "Overcoming finite-size effects in electronic structure simulations at extreme conditions," The Journal of Chemical Physics 154,144103 (2021).

[126] Markus Holzmann, Raymond C. Clay, Miguel A. Morales, Norm M. Tubman, David M. Ceperley, and Carlo Pierleoni, "Theory of finite size effects for electronic quantum Monte Carlo calculations of liquids and solids," Phys. Rev. B 94, 035126 (2016).

[127] Simone Chiesa, David M. Ceperley, Richard M. Martin, and Markus Holzmann, "Finite-size error in many-body simulations with long-range interactions," Phys. Rev. Lett. 97, 076404 (2006).

[128] N. D. Drummond, R. J. Needs, A. Sorouri, and W. M. C. Foulkes, "Finite-size errors in continuum quantum monte carlo calculations," Phys. Rev. B 78, 125106 (2008).

[129] Yu Shi, "Superfluidity or supersolidity as a consequence of off-diagonal long-range order," Phys. Rev. B 72, 014533 (2005).

[130] A. A. Kugler, "Bounds for some equilibrium properties of an electron gas," Phys. Rev. A 1, 1688 (1970).

[131] V. Bobrov, N. Klyuchnikov, and S Triger, "Exact relations for structure factor of a coulomb system," Teoreticheskaya i Matematicheskaya Fizika 89, 263-277 (1991).

[132] J. Vorberger and D. O. Gericke, "Ab initio approach to model x-ray diffraction in warm dense matter," Phys. Rev. E 91, 033112 (2015).

[133] S. H. Glenzer and R. Redmer, "X-ray thomson scattering in high energy density plasmas," Rev. Mod. Phys 81, 1625 (2009).

[134] U. Zastrau, P. Sperling, M. Harmand, A. Becker, T. Bornath, R. Bredow, S. Dziarzhytski, T. Fennel, L. B. Fletcher, E. F" orster, S. G"ode, G. Gregori, V. Hilbert, D. Hochhaus, B. Holst, T. Laarmann, H. J. Lee, T. Ma, J. P. Mithen, R. Mitzner, C. D. Murphy, M. Nakatsutsumi, P. Neumayer, A. Przystawik, S. Roling, M. Schulz, B. Siemer, S. Skruszewicz, J. Tiggesb"aumker, S. Toleikis, T. Tschentscher, T. White, M. W" ostmann, H. Zacharias, T. D" oppner, S. H. Glenzer, and R. Redmer, "Resolving ultrafast heating of dense cryogenic hydrogen," Phys. Rev. Lett 112, 105002 (2014).

[135] L. B. Fletcher, J. Vorberger, W. Schumaker, C. Ruyer, S. Goede, E. Galtier, U. Zastrau, E. P. Alves, S. D. Baalrud, R. A. Baggott, B. Barbrel, Z. Chen, T. Döppner, M. Gauthier, E. Granados, J. B. Kim, D. Kraus, H. J. Lee, M. J. MacDonald, R. Mishra, A. Pelka, A. Ravasio, C. Roedel, A. R. Fry, R. Redmer, F. Fiuza, D. O. Gericke, and S. H. Glenzer, "Electron-ion temperature relaxation in warm dense hydrogen observed with picosecond resolved x-ray scattering," Frontiers in Physics 10 (2022).

[136] Paul Hamann, Linda Kordts, Alexey Filinov, Michael Bonitz, Tobias Dornheim, and Jan Vorberger, "Prediction of a roton-type feature in warm dense hydrogen," Phys. Rev. Res. 5, 033039 (2023).

[137] Tobias Dornheim, Zhandos Moldabekov, Jan Vorberger, Hanno Kählert, and Michael Bonitz, "Electronic pair alignment and roton feature in the warm dense electron gas," Communications Physics 5, 304 (2022).

[138] Henri Godfrin, Matthias Meschke, Hans-Jochen Lauter, Ahmad Sultan, Helga M. Böhm, Eckhard Krotscheck, and Martin Panholzer, "Observation of a roton collective mode in a two-dimensional fermi liquid," Nature 483, 576-579 (2012).

[139] Viktor Bobrov, Sergey Trigger, and Daniel Litinski, "Universality of the phonon-roton spectrum in liquids and superfluidity of 4he," Zeitschrift für Naturforschung A 71, 565-575 (2016).

[140] Tobias Dornheim, Zhandos A. Moldabekov, Jan Vorberger, and Burkhard Militzer, "Path integral Monte Carlo approach to the structural properties and collective excitations of liquid ${ }^{3} \mathrm{He}$ without fixed nodes," Scientific Reports 12, 708 (2022).

[141] G. Ferré and J. Boronat, "Dynamic structure factor of liquid ${ }^{4} \mathrm{He}$ across the normal-superfluid transition," Phys. Rev. B 93, 104510 (2016).
[142] S. Mazevet, M. P. Desjarlais, L. A. Collins, J. D. Kress, and N. H. Magee, "Simulations of the optical properties of warm dense aluminum," Phys. Rev. E 71, 016409 (2005).

[143] M. P. Desjarlais, J. D. Kress, and L. A. Collins, "Electrical conductivity for warm, dense aluminum plasmas and liquids," Phys. Rev. E 66, 025401(R) (2002).

[144] Tobias Dornheim, Maximilian P. Böhme, Zhandos A. Moldabekov, and Jan Vorberger, "Electronic density response of warm dense hydrogen on the nanoscale," Phys. Rev. E 108, 035204 (2023).

[145] Zhandos A. Moldabekov, Michele Pavanello, Maximilian P. Böhme, Jan Vorberger, and Tobias Dornheim, "Linear-response time-dependent density functional theory approach to warm dense matter with adiabatic exchange-correlation kernels," Phys. Rev. Res. 5, 023089 (2023).

[146] Zhandos A. Moldabekov, Jan Vorberger, Mani Lokamani, and Tobias Dornheim, "Averaging over atom snapshots in linear-response TDDFT of disordered systems: A case study of warm dense hydrogen," The Journal of Chemical Physics 159, 014107 (2023).

[147] J. R. Rygg, P. M. Celliers, and G. W. Collins, "Specific heat of electron plasma waves," Phys. Rev. Lett. 130, 225101 (2023).

[148] C. Bowen, G. Sugiyama, and B. J. Alder, "Static dielectric response of the electron gas," Phys. Rev. B 50, 14838 (1994).

[149] Tobias Dornheim, Zhandos A. Moldabekov, and Jan Vorberger, "Nonlinear electronic density response of the ferromagnetic uniform electron gas at warm dense matter conditions," Contributions to Plasma Physics 61, e202100098 (2021).

[150] D. M. Ceperley, "Path-integral calculations of normal liquid ${ }^{3} \mathrm{He}$," Phys. Rev. Lett. 69, 331-334 (1992).

[151] C. Ebner and D. O. Edwards, "The low temperature thermodynamic properties of superfluid solutions of ${ }^{3} \mathrm{He}$ in ${ }^{4} \mathrm{He}, "$ Phys. Rep. 2, 77 (1970).

[152] E. Krotscheck and M. Saarela, "Theory of ${ }^{3} \mathrm{He}-{ }^{4} \mathrm{He}$ mixtures: energetics, structure, and stability," Phys. Rep. 232, 1 (1993).

[153] I. Ferrier-Barbut, M. Delehaye, S. Laurent, A. T. Grier, M. Pierce, B. S. Rem, F. Chevy, and C. Salomon, "A mixture of bose and fermi superfluids," Science 345, 1035 (2014).

[154] M. Boninsegni and D. M. Ceperley, "Path integral Monte Carlo simulation of isotopic liquid helium mixtures," Phys. Rev. Lett. 74, 2288 (1995).

[155] M. Boninsegni and S. Moroni, "Microscopic calculation of superfluidity and kinetic energies in isotopic liquid helium mixtures," Phys. Rev. Lett. 78, 1727 (1997).

[156] S. Moroni, S. Fantoni, and A. Fabrocini, "Deep inelastic response of liquid helium," Phys. Rev. B 58, 11607 (1998).

[157] S. Ujevic, V. Zampronio, B. R. de Abreu, and S. A. Vitiello, "Properties of fermionic systems with the pathintegral ground state method," SciPost Phys. Core 6, 031 (2023).

[158] M. Boninsegni, "Kinetic energy and momentum distribution of isotopic liquid helium mixtures," J. Chem. Phys. 148, 102308 (2018).


[^0]:    * t.dornheim@hzdr.de

</end of paper 2>


<paper 3>
# Ab initio Density Response and Local Field Factor of Warm Dense Hydrogen 

Tobias Dornheim, ${ }^{1,2, *}$ Sebastian Schwalbe, ${ }^{1,2}$ Panagiotis Tolias, ${ }^{3}$<br>Maximilian P. Böhme, ${ }^{1,2,4}$ Zhandos A. Moldabekov, ${ }^{1,2}$ and Jan Vorberger ${ }^{2}$<br>${ }^{1}$ Center for Advanced Systems Understanding (CASUS), D-02826 Görlitz, Germany<br>${ }^{2}$ Helmholtz-Zentrum Dresden-Rossendorf (HZDR), D-01328 Dresden, Germany<br>${ }^{3}$ Space and Plasma Physics, Royal Institute of Technology (KTH), Stockholm, SE-100 44, Sweden<br>${ }^{4}$ Technische Universität Dresden, D-01062 Dresden, Germany


#### Abstract

We present quasi-exact $a b$ initio path integral Monte Carlo (PIMC) results for the partial static density responses and local field factors of hydrogen in the warm dense matter regime, from solid density conditions to the strongly compressed case. The full dynamic treatment of electrons and protons on the same footing allows us to rigorously quantify both electronic and ionic exchange-correlation effects in the system, and to compare with earlier incomplete models such as the archetypal uniform electron gas [Phys. Rev. Lett. 125, 235001 (2020)] or electrons in a fixed ion snapshot potential [Phys. Rev. Lett. 129, 066402 (2022)] that do not take into account the interplay between the two constituents. The full electronic density response is highly sensitive to electronic localization around the ions, and our results constitute unambiguous predictions for upcoming X-ray Thomson scattering (XRTS) experiments with hydrogen jets and fusion plasmas. All PIMC results are made freely available and can directly be used for a gamut of applications, including inertial confinement fusion calculations and the modelling of dense astrophysical objects. Moreover, they constitute invaluable benchmark data for approximate but computationally less demanding approaches such as density functional theory or PIMC within the fixed-node approximation.


## I. INTRODUCTION

The properties of hydrogen at extreme temperatures, densities, and pressures are of paramount importance for a wealth of applications [1, 2]. An excellent example is given by inertial confinement fusion (ICF) [3], where the fuel capsule (most commonly a deuterium-tritium fuel) has to traverse such warm dense matter (WDM) conditions on its pathway towards ignition [4]. In several recent breakthrough experiments, it has been demonstrated at the National Ignition Facility (NIF) [5] that it is indeed possible to ignite the fuel [6] and even produce a net energy gain with respect to the laser energy deposited in the capsule $[7,8]$. While undoubtedly a number of challenges remain on the way to an operational fusion power plant [9], there is a broad consensus that ICF has great potential to emerge as a future source of safe and sustainable energy. A second class of applications is given by astrophysical objects, such as giant planet interiors [1, 10-12] (e.g., Jupiter in our solar system, but also exoplanets) and brown dwarfs [13, 14]. Here, the properties of hydrogen are of key relevance to describe the evolution of these objects, and to understand observations such as the gravitational moments of Jupiter $[15-17]$.

Despite its apparent simplicity, a rigorous theoretical description of hydrogen remains notoriously elusive over substantial parts of the relevant phase diagram [18-20]. A case in point is given by the insulator-to-metal phase transition, which is highly controversial both from a theoretical [18, 21-23] and an experimental [24-26] perspective. The situation is hardly better within the WDM[^0]

regime that is of particular relevance for both ICF and astrophysical applications $[1,27]$.

Specifically, the WDM regime is usually defined in terms of the Wigner-Seitz radius $r_{s}=\left(3 / 4 \pi n_{e}\right)^{1 / 3}$ (with $n_{e}=N_{e} / \Omega$ the electron density) and the degeneracy temperature $\Theta=k_{\mathrm{B}} T / E_{\mathrm{F}}$ (with $E_{\mathrm{F}}$ the Fermi energy of an electron gas at equal density [28]), both of the order of unity [29], i.e., $r_{s} \sim \Theta \sim 1$. Such absence of small parameters necessitates a full treatment of the complex interplay between strong thermal excitations (generally ruling out electronic ground-state methods), Coulomb coupling between both electrons and ions (often ruling out weak-coupling expansions such as Green functions [30]), and quantum effects such as Pauli blocking and diffraction (precluding semi-classical schemes such as molecular dynamics simulations with effective quantum potentials [31]). Consequently, there exists no single method that is reliable over the entire WDM regime [27]. In practice, a widely used method is given by a combination of molecular dynamics (MD) simulations of the ions with a thermal density functional theory (DFT) [32] treatment of the degenerate electrons based on the BornOppenheimer approximation. On the one hand, such DFT-MD simulations are often computationally manageable and, in principle, provide access to various material properties including the equation-of-state [33-36], linear response functions [37-39], and the electrical conductivity [40-43]. On the other hand, the DFT accuracy strongly depends on the employed exchange-correlation (XC) functional $[18,44]$, which has to be supplied as an external input. While the accuracy of many functionals is reasonably well understood in the ground state [45], the development of novel, thermal XC-functionals that are suitable for application at extreme temperatures has only started very recently [43, 46-49]. Moreover, many calcu-
lations require additional input, such as the XC-kernel for linear-response time-dependent DFT (LR-TDDFT) calculations of WDM and beyond [50].

Indeed, the linear density response of a system to an external perturbation [19] constitutes a highly important class of material properties in the context of WDM research. Such linear-response theory (LRT) properties are probed, e.g., in X-ray Thomson scattering (XRTS) experiments $[51,52]$, which have emerged as a key diagnostic of WDM [53, 54]. In principle, the measured XRTS intensity gives one access to the equation-of-state properties of the probed material [55-57], which is of paramount importance for ICF applications and the modelling of astrophysical objects. Unfortunately, in practice, the interpretation of the XRTS intensity is usually based on uncontrolled approximations such as the widely used Chihara decomposition [58] into effectively bound and free electrons. Thus, the quality of inferred parameters such as temperature, density, or ionization remains generally unclear.

LRT properties are also ubiquitous throughout WDM theory and central to the calculation of stopping power and electronic friction properties [59, 60], electrical and thermal conductivities $[61,62]$, opacity [63, 64], ionization potential depression [65], effective ion-ion potentials [66, 67], as well as the construction of advanced nonlocal XC-functionals for thermal DFT simulations $[68,69]$.

Owing to its key role, the electronic linear density response has been extensively studied for the simplified uniform electron gas (UEG) model [28, 70], where ions are treated as a rigid homogeneous charge-neutralizing background; see Ref. [19] and the references therein. These efforts have culminated in different parametrizations of the static local field factor [71-73], which is formally equivalent to the aforementioned XC-kernel - a basic input for LR-TDDFT simulations and other applications. Very recently, Böhme et al. $[74,75]$ have extended these efforts and obtained quasi-exact $a b$ initio path integral Monte Carlo (PIMC) [76-78] results for the density response of an electron gas in the external potential of a fixed proton snapshot. On the one hand, these results are directly comparable with thermal DFT simulations of hydrogen, which has given important insights into the accuracy of various XC-functionals [50, 79, 80]. On the other hand, these calculations are computationally very inefficientrequiring a large set of independent simulations to estimate the density response at a single wavenumber for a given snapshot-and, more importantly, miss the important dynamic interplay between electrons and protons.

In the present work, we overcome these fundamental limitations by presenting the first $a b$ initio PIMC results for the linear density response of full two-component warm dense hydrogen. By treating the electrons and protons dynamically on the same level (i.e., without invoking the Born Oppenheimer approximation), we access all components of the density response, including the electron-electron, electron-proton, and proton-proton local field factors (i.e., XC-kernels). Indeed, a consistent treatment of the interaction between the electrons and ions is crucial to capture the effects of electronic localization around the protons, which has important implications for the interpretation of XRTS experiments. These effects are particularly important for solid state densities $\left(r_{s}=3.23\right)$, where they are significant over the entire relevant wavenumber range. They are also substantial for metallic densities $\left(r_{s}=2\right)$ and even manifest at strong compression $\left(r_{s}=1\right)$ for small wave numbers.

Our simulations are quasi-exact; no fixed-node approximation [81] is imposed. To deal with the fermion sign problem [82, 83] - an exponential computational bottleneck in quantum Monte Carlo simulations of degenerate Fermi systems - we average over a large number of Monte Carlo samples, making our simulations computationally very expensive; the cost of this study is estimated to be $\mathcal{O}\left(10^{7}\right)$ CPUh. Additionally, we employ the recently introduced $\xi$-extrapolation method [84-88] to access larger system sizes. We find that finite-size effects are generally negligible at these conditions, which is consistent with previous results for the UEG model [70, 89, 90].

We are convinced that our results will open up multiple opportunities for future research. First, the quasiexact nature of our results makes them a rigorous benchmark for computationally less costly but approximate approaches such as thermal DFT or PIMC within the fixednode approximation. Moreover, the obtained LRT properties constitute direct predictions for upcoming XRTS experiments with hydrogen jets [91, 92] and ICF plasmas [5]. This makes them also ideally suited to benchmark Chihara models $[54,58,93]$ that are commonly used to diagnose XRTS measurements and infer an equationof-state. Of particular value are our results for the various partial local field factors, which can be used as input for innumerable applications such as transport property estimates or the construction of novel XC-functionals for thermal DFT simulations. Finally, we note that the presented study of the density response of warm dense hydrogen is interesting in its own right, and gives us new insights into the interplay of electronic localization, quantum effects, and electron-ion correlations in the WDM regime.

The paper is organized as follows: In Sec. II, we provide the relevant theoretical background; a brief introduction of the PIMC method and its estimate of imaginary-time correlation functions (II B), the full hydrogen Hamiltonian (II A), the linear density response theory (II C), and its relation to XRTS experiments (II D). In Sec. III, we present our extensive novel simulation results covering the cases of metallic density (III A), solid state density (III B), and strong compression (III C). The paper is concluded by a summary and outlook in Sec. IV. Additional technical details are provided in the appendices.

## II. THEORY

## A. Hamiltonian

We assume Hartree atomic units (i.e., $\hbar=m_{e}=e=1$ ) throughout this work, unless otherwise specified. The full Hamiltonian governing the behaviour of hydrogen is given by $\qquad$

$$
\begin{equation*}
\hat{H}_{H}=-\frac{1}{2} \sum_{l=1}^{N} \nabla_{l, e}^{2}-\frac{1}{2 m_{p}} \sum_{l=1}^{N} \nabla_{l, p}^{2}+\sum_{\substack{k=1 \\ l<k}}^{N} \sum_{l=1}^{N} W_{\mathrm{E}}\left(\hat{r}_{l}, \hat{r}_{k}\right)+\sum_{\substack{k=1 \\ l<k}}^{N} \sum_{l=1}^{N} W_{\mathrm{E}}\left(\hat{I}_{l}, \hat{I}_{k}\right)-\sum_{k=1}^{N} \sum_{l=1}^{N} W_{\mathrm{E}}\left(\hat{I}_{l}, \hat{r}_{k}\right) \tag{1}
\end{equation*}
$$

where the first and second term correspond to the kinetic energy of the electrons and protons. The pair interaction $W_{\mathrm{E}}$ is given by the usual Ewald summation, and we follow the conventions introduced in Ref. [94]; $\hat{r}$ and $\hat{I}$ denote the position operators of electrons and protons.

## B. Path integral Monte Carlo

The ab initio PIMC method [76, 77, 95] constitutes one of the most successful tools for the description of interacting, quantum degenerate many-body systems at finite temperature. The central property is given by the canonical (i.e., inverse temperature $\beta=1 / T$, volume $\Omega=$ $L^{3}$, the simulation box length $L$, and number density $n=N / \Omega$ are fixed) partition function evaluated in the coordinate representation

$$
\begin{equation*}
Z_{N, \Omega, \beta}=\frac{1}{N_{\uparrow}!N_{\downarrow}!} \sum_{\sigma_{N_{\uparrow}} \in S_{N_{\uparrow}}} \sum_{\sigma_{N_{\downarrow}} \in S_{N_{\downarrow}}} \xi^{N_{p p}} \int \mathrm{d} \mathbf{R}\left\langle\mathbf{R}\left|e^{-\beta \hat{H}_{H}}\right| \hat{\pi}_{\sigma_{N_{\uparrow}}} \hat{\pi}_{\sigma_{N_{\downarrow}}} \mathbf{R}\right\rangle \tag{2}
\end{equation*}
$$

where the meta-variable $\mathbf{R}=\left(\mathbf{r}_{1}, \ldots, \mathbf{r}_{N}, \mathbf{I}_{1}, \ldots, \mathbf{I}_{N}\right)^{T}$ contains the coordinates of both electrons $\left(\mathbf{r}_{l}\right.$ ) and protons $\left(\mathbf{I}_{l}\right)$. Throughout this work, we consider the fully unpolarized case with an equal number of spin-up and spin-down electrons, $N_{\uparrow}=N_{\downarrow}=N / 2$. Eq. (2) contains a summation over all possible permutations $\sigma_{N_{i}}$ of electrons of the same spin orientation $i \in\{\uparrow, \downarrow\}$, which are realized by the corresponding permutation operator $\hat{\pi}_{\sigma_{N_{i}}}$; $N_{p p}$ denotes the total number of pair permutations of both spin-up and spin-down electrons. Note that we treat the heavier protons as distinguishable quantum particles (sometimes referred to as boltzmannons in the literature), which is exact at the investigated conditions. In fact, even protonic quantum delocalization effects are only of the order of $0.1 \%$ here, cf. Figs. 3, 9, 13 below. The variable $\xi$ determines the type of quantum statistics of the electrons with $\xi=-1, \xi=0$, and $\xi=1$ corresponding to Fermi-Dirac, Boltzmann-Maxwell, and Bose-Einstein statistics, respectively. Only $\xi=-1$ has distinct physical meaning for warm dense hydrogen, although other values can give valuable insights into the importance of quantum degeneracy effects for different observables [87, 88].

A detailed derivation of the PIMC method is beyond our scope and has been presented elsewhere $[75,77,78]$.
In essence, the complicated quantum many-body problem of interest, as defined by Eq. (2), is mapped onto an effectively classical system of interacting ring polymers with $P$ segments each. This is the celebrated classical isomorphism [96]. A schematic illustration of this idea is presented in Fig. 1, where we show a configuration of $N=3$ hydrogen atoms in the $\tau$ - $x$-plane. In the path integral picture, each particle is represented by an entire path of particle coordinates along the imaginary-time $\tau$ with $P$ imaginary-time slices. The filled and empty circles correspond to the coordinates of electrons and ions, and beads of the same particle on adjacent time slices are connected by a harmonic spring potential. The latter are depicted by the red and green lines for electrons and protons. The extension of these paths corresponds to the thermal wavelength $\lambda_{T}^{a}=\sqrt{2 \pi \beta / m_{a}}$, that is smaller by a factor of $1 / \sqrt{m_{p}} \approx 0.023$ for the protons. The basic idea of the PIMC method is to use the Metropolis algorithm [97] to generate a Markov chain of such path configurations $\mathbf{X}$, where the meta-variable $\mathbf{X}$ contains all coordinates and also the complete information about the sampled permutation structure. A first obstacle is given by the diverging Coulomb attraction, which prevents the straightforward application of the Trotter formula [98].

![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-04.jpg?height=732&width=851&top_left_y=179&top_left_x=190)

FIG. 1. A schematic illustration of the PIMC representation. Shown is a configuration of $N=3$ hydrogen atoms in the $x$ $\tau$-plane for $P=6$ imaginary-time slices. Each electron (filled circles) and each proton (empty circles) is represented by a path along the imaginary-time axis $\tau$ of $P$ coordinates, which are effectively connected by harmonic spring potentials (red and green connections); the extension of the paths is directly related to the respective thermal wavelength $\lambda_{T}^{a}=\sqrt{2 \pi \beta / m_{a}}$. The yellow horizontal lines illustrate the evaluation of the density of species $a$ and $b$ in reciprocal space at an imaginarytime distance $\tau$ to estimate the ITCF $F_{a b}(\mathbf{q}, \tau)$ [Eq. (3)].

To avoid the associated path collapse, we employ the well-known pair approximation [75, 77, 99], which effectively leads to an off-diagonal quantum potential that remains finite even at zero distance. An additional difficulty arises from the factor $(-1)^{N_{p p}}$ in the case of Fermi statistics. For every change in the permutation structure, the sign of the configuration weight changes. For example, we show a configuration with $N_{p p}=1$ pair exchange in Fig. 1 (i.e., the two electrons on the right), which has a negative configuration weight. In practice, the resulting cancellation of positive and negative terms leads to an exponential increase in the compute time towards low temperatures and large numbers of electrons $N$; this bottleneck is known as the fermion sign problem in the literature [82, 83]; see also Appendix A for additional details. While different, often approximate strategies to mitigate the sign problem have been suggested [81, 100-104], here, we carry out exact direct PIMC solutions that are subject to the full amount of cancellations. The comparably large noise-to-signal level is reduced by averaging over a large number of independent Monte Carlo samples, and our results are exact within the given error bars.

An additional advantage of the direct PIMC method concerns its straightforward access to various imaginarytime correlation functions (ITCF) [78, 105-107]. In the context of the present work, the focus lies on the densitydensity ITCF (hereafter shortly referred to as ITCF)

$$
\begin{equation*}
F_{a b}(\mathbf{q}, \tau)=\left\langle\hat{n}_{a}(\mathbf{q}, 0) \hat{n}_{b}(-\mathbf{q}, \tau)\right\rangle \tag{3}
\end{equation*}
$$

which measures the decay of correlations between particles of species $a$ and $b$ along the imaginary-time diffusion process, see yellow lines and symbols in Fig. 1. As we shall see in Sec. II C, the ITCF gives direct access to the static density response of warm dense hydrogen from a single simulation of the unperturbed system. The ITCF is interesting in its own right [108, 109], and can be measured in XRTS experiments $[53,110,111]$, see Sec. II D.

## C. Partial density response functions and partial local field factors

The basic idea of the density response formalism is to study a system that is subject to a weak external perturbation which couples linearly to the Hamiltonian [19]:

$$
\begin{align*}
\hat{H}_{\mathbf{q}, \omega ; A_{e}, A_{p}}=\hat{H}_{H} & +2 A_{e} \sum_{l=1}^{N} \cos \left(\mathbf{q} \cdot \hat{\mathbf{r}}_{l}-\omega t\right)  \tag{4}\\
& +2 A_{p} \sum_{l=1}^{N} \cos \left(\mathbf{q} \cdot \hat{\mathbf{I}}_{l}-\omega t\right)
\end{align*}
$$

Here $\hat{H}_{H}$ denotes the unperturbed Hamiltonian of the full hydrogen system [Eq. (1)], and $\mathbf{q}$ and $\omega$ are the wavevector and frequency of the external monochromatic perturbation. It is noted that we distinguish the perturbation amplitude of the electrons $\left(A_{e}\right)$ and protons $\left(A_{p}\right)$, which allows us to study different species-resolved components of the density response, as we shall discuss in detail. In the limit of an infinitesimally small external perturbation strength, the induced density is a linear function of $A_{a}$ and linear response theory is applicable [19,112]:

$$
\begin{equation*}
\delta\left\langle\hat{\rho}_{a}(\mathbf{q}, \omega)\right\rangle=\chi_{a a}^{(0)}(\mathbf{q}, \omega)\left(A_{a}+\sum_{b} v_{a b}(\mathbf{q})\left[1-G_{a b}(\mathbf{q}, \omega)\right] \delta\left\langle\hat{\rho}_{b}(\mathbf{q}, \omega)\right\rangle\right) \tag{5}
\end{equation*}
$$

where the induced density perturbation $\delta\left\langle\hat{\rho}_{a}\right\rangle$ is evaluated in the reciprocal space and where $\chi_{a a}^{(0)}(\mathbf{q}, \omega)$ denotes the linear density response of a noninteracting system (of species $a$ ) at the same conditions. Further, the complete information about species-resolved exchange-correlation effects is contained in the local field factors $G_{a b}(\mathbf{q}, \omega)$.

Eq. (5) defines a coupled system of equations that explicitly depends on the perturbation $A_{a}$ of all components.

The fundamental property in linear density response theory is the linear density response function $\chi_{a b}(\mathbf{q}, \omega)$ that describes the density response of the species $a$ to a potential energy perturbation applied to the species $b$,

$$
\begin{equation*}
\delta\left\langle\hat{\rho}_{a}(\mathbf{q}, \omega)\right\rangle=A_{b} \chi_{a b}(\mathbf{q}, \omega) \tag{6}
\end{equation*}
$$

This dynamic susceptibility constitutes a material property of the unperturbed system, and is directly related to the dielectric function $[113,114]$. Moreover, it is of central importance to LR-TDDFT simulations [115], and for the interpretation of XRTS experiments with WDM and beyond, as we explain in Sec. IID below.

Let us focus on the static limit $\chi_{a b}(\mathbf{q}, 0) \equiv \chi_{a b}(\mathbf{q})$ that describes the response to a time-independent cosinusoidal perturbation. Moroni and collaborators [116, 117] have presented the first accurate results for the linear static density response function and the local field factor of the ground state UEG based on diffusion QMC simulations. This idea was subsequently adapted to PIMC and configuration PIMC (CPIMC) simulations of the UEG in the WDM regime $[118,119]$. More specifically, the basic idea is to utilize the formal perturbation strength expansion of non-linear density response theory that reads

$$
\begin{equation*}
\delta\left\langle\hat{\rho}_{a}(\mathbf{q})\right\rangle=\chi_{a b}^{(1,1)}(\mathbf{q}) A_{b}+\chi_{a b}^{(1,3)}(\mathbf{q}) A_{b}^{3}+\mathcal{O}\left(A_{b}^{5}\right) \tag{7}
\end{equation*}
$$

with $\chi_{a b}^{(m, l)}(\boldsymbol{q})$ the partial static density response of the order $l$ at a harmonic $m$, where $\chi_{a b}^{(1,1)}(\boldsymbol{q}) \equiv \chi_{a b}(\boldsymbol{q})$ [120]. This is a particular case of a general result which states that the total induced density at odd (even) harmonics is given by an infinite series of all $l \geq m$ odd (even) powers of the perturbation strength with the coefficients equal to $\chi_{a b}^{(m, l)}(\boldsymbol{q})[120]$. Thus, the induced density is estimated for various $A_{b}$ and a polynomial expansion $[19,116,121]$

$$
\begin{equation*}
\delta\left\langle\hat{\rho}_{a}(\mathbf{q})\right\rangle=c_{1} A_{b}+c_{3} A_{b}^{3}+\mathcal{O}\left(A_{b}^{5}\right) \tag{8}
\end{equation*}
$$

allows to identify the linear and cubic density response functions at the first harmonic (i.e., at the wavenumber of the perturbation) via the correspondence $c_{1}=\chi_{a b}(\mathbf{q})$ and $c_{3}=\chi_{a b}^{(1,3)}(\mathbf{q})$. On the one hand, this direct perturbation method is formally exact, and can easily be adapted to other methods such as DFT [50, 122, 123]. On the other hand, it is computationally very expensive, since independent simulations need to be carried out for multiple perturbation amplitudes $A_{b}$ just to extract the response $\chi_{a b}(\mathbf{q})$ for a single wavevector at a given density and temperature combination. An elegant alternative is given by the imaginary-time version of the fluctuationdissipation theorem [108], whose species-resolved version reads

$$
\begin{equation*}
\chi_{a b}(\mathbf{q}, 0)=-\frac{\sqrt{N_{a} N_{b}}}{\Omega} \int_{0}^{\beta} \mathrm{d} \tau F_{a b}(\mathbf{q}, \tau) \tag{9}
\end{equation*}
$$

which implies that the species-resolved density responses at all accessible wavevectors can be extracted from a single simulation of the unperturbed system by estimating the corresponding partial $F_{a b}(\mathbf{q}, \tau)$. This method has been extensively applied to study the density response of the UEG at finite temperatures over a vast range of parameters [71, 124-130]. Here, we employ this second route for the bulk of our new results, while we use the direct perturbation route as an independent cross check.

Let us next consider the local field factors $G_{a b}(\mathbf{q}, \omega)$ in more detail. For the UEG, the polarization potential approach leads to the well-known linear density response expression $[28,70,131,132]$

$$
\begin{equation*}
\chi_{e e}(\mathbf{q}, \omega)=\frac{\chi_{e e}^{(0)}(\mathbf{q}, \omega)}{1-v_{e e}(q)\left[1-G_{e e}(\mathbf{q}, \omega)\right] \chi_{e e}^{(0)}(\mathbf{q}, \omega)} \tag{10}
\end{equation*}
$$

Setting $G_{e e}(\mathbf{q}, \omega) \equiv 0$ in Eq. (10) leads to the random phase approximation (RPA) that describes the electronic density response on a mean-field level. It is pointed out that spin-resolved UEG generalizations have been presented in the literature $[112,133]$, where the spin-up and spin-down electrons are treated as two distinct species.

For multi-component systems, the coupling between the different species makes the situation considerably more complicated. We note that the terms partial and species-resolved are used interchangeably throughout the text for the associated LRT quantities of multicomponent systems. Introducing the so-called vertex corrected interaction

$$
\begin{equation*}
\theta_{a b}(\boldsymbol{q}, \omega)=v_{a b}(\boldsymbol{q})\left[1-G_{a b}(\boldsymbol{q}, \omega)\right] \tag{11}
\end{equation*}
$$

the polarization potential approach leads to the following expression for the linear density perturbation $\delta n$ that is induced by the potential energy perturbation $\delta U$ [134]

$$
\begin{equation*}
\delta n_{a}(\boldsymbol{q}, \omega)=\chi_{a a}^{(0)}(\boldsymbol{q}, \omega) \delta U_{a}(\boldsymbol{q}, \omega)+\sum_{c} \chi_{a a}^{(0)}(\boldsymbol{q}, \omega) \theta_{a c}(\boldsymbol{q}, \omega) \delta n_{c}(\boldsymbol{q}, \omega) \tag{12}
\end{equation*}
$$

which clearly constitutes the generalized version of Eq. (5) for non-monochromatic perturbations. The func- tional derivative definition of the partial density response function $\chi_{a b}(\boldsymbol{q}, \omega)=\delta n_{a}(\boldsymbol{q}, \omega) / \delta U_{b}(\boldsymbol{q}, \omega)$ and the identity $\delta U_{a}(\boldsymbol{q}, \omega) / \delta U_{b}(\boldsymbol{q}, \omega)=\delta_{a b}$ yield

$$
\begin{equation*}
\chi_{a b}(\boldsymbol{q}, \omega)=\chi_{a a}^{(0)}(\boldsymbol{q}, \omega) \delta_{a b}+\sum_{c} \chi_{a a}^{(0)}(\boldsymbol{q}, \omega) \theta_{a c}(\boldsymbol{q}, \omega) \chi_{c b}(\boldsymbol{q}, \omega) \tag{13}
\end{equation*}
$$

The ideal density response functions $\chi_{a b}^{(0)}$ are nonzero only if $a=b$. Note the reciprocal connections, i.e., $\chi_{a b}(\boldsymbol{q}, \omega)=\chi_{b a}(\boldsymbol{q}, \omega), \theta_{a b}(\boldsymbol{q}, \omega)=\theta_{b a}(\boldsymbol{q}, \omega), G_{a b}(\boldsymbol{q}, \omega)=$ $G_{b a}(\boldsymbol{q}, \omega)$, that are a consequence of Newton's third law $v_{a b}(\boldsymbol{q})=v_{b a}(\boldsymbol{q})$. The above $3 \times 3$ set of linear equations can be explicitly solved for the partial density response functions in dependence of $\theta_{a b}(\boldsymbol{q}, \omega)$. For hydrogen, $a, b, c=\{e, p\}$, this leads to

$\chi_{e e}(\boldsymbol{q}, \omega)=\frac{1}{\Delta(\boldsymbol{q}, \omega)} \chi_{e e}^{(0)}(\boldsymbol{q}, \omega)\left[1-\theta_{p p}(\boldsymbol{q}, \omega) \chi_{p p}^{(0)}(\boldsymbol{q}, \omega)\right]$,

$\chi_{p p}(\boldsymbol{q}, \omega)=\frac{1}{\Delta(\boldsymbol{q}, \omega)} \chi_{p p}^{(0)}(\boldsymbol{q}, \omega)\left[1-\theta_{e e}(\boldsymbol{q}, \omega) \chi_{e e}^{(0)}(\boldsymbol{q}, \omega)\right]$,

$\chi_{e p}(\boldsymbol{q}, \omega)=\frac{1}{\Delta(\boldsymbol{q}, \omega)}\left[\chi_{e e}^{(0)}(\boldsymbol{q}, \omega) \theta_{e p}(\boldsymbol{q}, \omega) \chi_{p p}^{(0)}(\boldsymbol{q}, \omega)\right]$,

where the auxiliary response $\Delta(\boldsymbol{q}, \omega)$ is the determinant that is given by

$$
\begin{aligned}
\Delta(\boldsymbol{q}, \omega)= & {\left[1-\theta_{e e}(\boldsymbol{q}, \omega) \chi_{e e}^{(0)}(\boldsymbol{q}, \omega)\right]\left[1-\theta_{p p}(\boldsymbol{q}, \omega) \chi_{p p}^{(0)}(\boldsymbol{q}, \omega)\right] } \\
& -\theta_{e p}^{2}(\boldsymbol{q}, \omega) \chi_{e e}^{(0)}(\boldsymbol{q}, \omega) \chi_{p p}^{(0)}(\boldsymbol{q}, \omega)
\end{aligned}
$$

When first principle results are available for $\chi_{a b}(\boldsymbol{q}, \omega)$, as in our case for the static limit of $\omega=0$, the above $3 \times 3$ set of linear equations can be explicitly solved for the hydrogen partial local field factors.

$$
\begin{align*}
& \theta_{e e}(\boldsymbol{q}, \omega)=\frac{1}{\chi_{e e}^{(0)}(\boldsymbol{q}, \omega)}-\frac{\chi_{p p}(\boldsymbol{q}, \omega)}{\chi_{e e}(\boldsymbol{q}, \omega) \chi_{p p}(\boldsymbol{q}, \omega)-\chi_{e p}^{2}(\boldsymbol{q}, \omega)}  \tag{14}\\
& \theta_{p p}(\boldsymbol{q}, \omega)=\frac{1}{\chi_{p p}^{(0)}(\boldsymbol{q}, \omega)}-\frac{\chi_{e e}(\boldsymbol{q}, \omega)}{\chi_{e e}(\boldsymbol{q}, \omega) \chi_{p p}(\boldsymbol{q}, \omega)-\chi_{e p}^{2}(\boldsymbol{q}, \omega)}  \tag{15}\\
& \theta_{e p}(\boldsymbol{q}, \omega)=\frac{\chi_{e p}(\boldsymbol{q}, \omega)}{\chi_{e e}(\boldsymbol{q}, \omega) \chi_{p p}(\boldsymbol{q}, \omega)-\chi_{e p}^{2}(\boldsymbol{q}, \omega)} \tag{16}
\end{align*}
$$

The limit of weak electron-proton coupling is expressed by $\chi_{e e}(\boldsymbol{q}, \omega) \chi_{p p}(\boldsymbol{q}, \omega) \gg \chi_{e p}^{2}(\boldsymbol{q}, \omega)$ which returns the onecomponent expressions

$$
\begin{aligned}
& \theta_{e e}(\boldsymbol{q}, \omega)=\frac{1}{\chi_{e e}^{(0)}(\boldsymbol{q}, \omega)}-\frac{1}{\chi_{e e}(\boldsymbol{q}, \omega)} \\
& \theta_{p p}(\boldsymbol{q}, \omega)=\frac{1}{\chi_{p p}^{(0)}(\boldsymbol{q}, \omega)}-\frac{1}{\chi_{p p}(\boldsymbol{q}, \omega)}
\end{aligned}
$$

The opposite limit of strong electron-proton coupling is expressed by $\chi_{e e}(\boldsymbol{q}, \omega) \chi_{p p}(\boldsymbol{q}, \omega) \ll \chi_{e p}^{2}(\boldsymbol{q}, \omega)$ which again returns the one-component expression but in absence of a non-interacting contribution

$$
\theta_{e p}(\boldsymbol{q}, \omega)=-\frac{1}{\chi_{e p}(\boldsymbol{q}, \omega)}
$$

## D. Connection to XRTS experiments

The measured intensity in an XRTS experiment can be expressed as $[19,51,110]$

$$
\begin{equation*}
I(\mathbf{q}, \omega)=S_{e e}(\mathbf{q}, \omega) \otimes R(\omega) \tag{17}
\end{equation*}
$$

where $S_{e e}(\mathbf{q}, \omega)$ is the dynamic structure factor (DSF) and $R(\omega)$ the combined source and instrument function. In practice, $R(\omega)$ is often known with sufficient accuracy from source monitoring, or from the characterization of a utilized backlighter X-ray source [135]. We note that a direct deconvolution to solve Eq. (17) is generally not stable due to noise in the experimental data; a model for $S_{e e}(\mathbf{q}, \omega)$ is usually convolved with $R(\omega)$ and compared with $I(\mathbf{q}, \omega)$ instead. The connection between $S_{e e}(\mathbf{q}, \omega)$ and the dynamic density response function introduced in the previous section is given by the wellknown fluctuation-dissipation theorem [28]

$$
\begin{equation*}
S_{e e}(\mathbf{q}, \omega)=-\frac{\operatorname{Im}\left\{\chi_{e e}(\mathbf{q}, \omega)\right\}}{\pi n_{e}\left(1-e^{-\beta \omega}\right)} \tag{18}
\end{equation*}
$$

In combination with the Kramers-Kronig causality relation that connects $\operatorname{Im}\left\{\chi_{e e}(\mathbf{q}, \omega)\right\}$ and $\operatorname{Re}\left\{\chi_{e e}(\mathbf{q}, \omega)\right\}[28]$, the fluctuation-dissipation theorem directly implies that $\chi_{e e}(\mathbf{q}, \omega)$ and $S_{e e}(\mathbf{q}, \omega)$ contain exactly the same information. Moreover, the static density response function $\chi_{e e}(\mathbf{q})$ is related to the DSF via the inverse frequency moment sum rule [136]

$$
\begin{equation*}
\chi_{e e}(\mathbf{q})=-2 n_{e} \int_{-\infty}^{\infty} \mathrm{d} \omega \frac{S_{e e}(\mathbf{q}, \omega)}{\omega} \tag{19}
\end{equation*}
$$

Therefore, accurate knowledge of $\chi_{e e}(\mathbf{q})$ gives direct insights into the low-frequency behaviour of $S_{e e}(\mathbf{q}, \omega)$ that is dominated by electron-proton coupling effects such as localization and ionization. Yet, the direct evaluation of the RHS. of Eq. (19) based on experimental XRTS data is generally prevented by the convolution with $R(\omega)$, see Eq. (17). In practice, this problem can be circumvented elegantly by considering the relation between $S_{e e}(\mathbf{q}, \omega)$ and the ITCF $F_{e e}(\mathbf{q}, \tau)$,

$$
\begin{equation*}
F_{e e}(\mathbf{q}, \tau)=\mathcal{L}\left[S_{e e}(\mathbf{q}, \omega)\right]=\int_{-\infty}^{\infty} \mathrm{d} \omega S_{e e}(\mathbf{q}, \omega) e^{-\tau \omega} \tag{20}
\end{equation*}
$$

where $\mathcal{L}[\ldots]$ denotes the two-sided Laplace transform operator. Making use of the convolution theorem

$$
\begin{equation*}
\mathcal{L}\left[S_{e e}(\mathbf{q}, \omega)\right]=\frac{\mathcal{L}\left[S_{e e}(\mathbf{q}, \omega) \otimes R(\omega)\right]}{\mathcal{L}[R(\omega)]} \tag{21}
\end{equation*}
$$

one can thus deconvolve the measured scattering intensity in the imaginary-time domain. In fact, the evaluation of Eq. (21) turns out to be remarkably stable with
respect to the experimental noise $[53,110,111]$ and, thus, gives one direct access to the ITCF from an XRTS measurement. It is then straightforward to obtain the static electron-electron density response function $\chi_{e e}(\mathbf{q})$ from the experimentally measured ITCF via Eq. (9), which amounts to the imaginary time analogue of Eq. (19). Our new results for the static density response of full hydrogen thus constitute an unambiguous prediction for upcoming XRTS experiments with hydrogen jets, fusion plasmas, etc.

To get additional insights into the physics implications of $\chi_{e e}(\mathbf{q})$, we consider the widely used Chihara decomposition of the DSF [54, 58, 93],

$$
\begin{equation*}
S_{e e}(\mathbf{q}, \omega)=S_{\mathrm{el}}(\mathbf{q}, \omega)+\underbrace{S_{\mathrm{bf}}(\mathbf{q}, \omega)+S_{\mathrm{ff}}(\mathbf{q}, \omega)}_{S_{\mathrm{inel}}(\mathbf{q}, \omega)} \tag{22}
\end{equation*}
$$

The basic idea of Eq. (22) is to divide the electrons into effectively bound and free populations. The first contribution to the full DSF is given by the pseudo-elastic component

$$
\begin{equation*}
S_{\mathrm{el}}(\mathbf{q}, \omega)=W_{R}(\mathbf{q}) \delta(\omega) \tag{23}
\end{equation*}
$$

where the Rayleigh weight contains both the atomic form factor of bound electrons and a screening cloud of free electrons [137]. Therefore, this term originates exclusively from the electronic localization around the protons. The second contribution to the full DSF contains all inelastic contributions, i.e., transitions between bound and free electronic states (and the reverse process, free-bound transitions [93]) described by $S_{\mathrm{bf}}(\mathbf{q}, \omega)$ as well as transitions between free states described by $S_{\mathrm{ff}}(\mathbf{q}, \omega)$. It is important to note that the decomposition between bound and free states is arbitrary in practice and breaks down at high compression where even the orbitals of bound electrons overlap [138]; the PIMC method that we use in the present work does not distinguish between bound and free electrons and, as a consequence, is not afflicted by these problems. Nevertheless, the simple picture of the Chihara model Eq. (22) gives important qualitative insights into the density response of the system. For example, Eq. (19) directly implies that $\chi_{e e}(\mathbf{q})$ is highly sensitive to electronic localization around the ions, which should result in an increased density response of hydrogen compared to the UEG model [71, 72]; this is indeed what we infer from our PIMC results, as we shall see in Sec. III below.

## III. RESULTS

We use the extended ensemble sampling scheme from Ref. [139] that is implemented in the imaginary-time stochastic high-performance tool for ab initio research (ISHTAR) code [140], which is a canonical adaption of the worm algorithm by Boninsegni et al. [78, 141]. All results are freely available online [142] and can be used as input for other calculations, or as a rigorous benchmark for other methods. A short discussion of the $\xi$-extrapolation method [84, 86], used to simulate larger systems, is given in Appendix A.

## A. Metallic density: $r_{s}=2$

In this section, we investigate in detail the linear density response of hydrogen at a metallic density $r_{s}=2$ and at the electronic Fermi temperature $\Theta=1$. We begin our analysis with a study of the ITCF $F_{a b}(\mathbf{q}, \tau)$, which constitutes the basis for a significant part of the present work. In Fig. 2, we illustrate $F_{e e}(\mathbf{q}, \tau)$ (left), $F_{p p}(\mathbf{q}, \tau)$ (center), and $F_{e p}(\mathbf{q}, \tau)$ (right) in the relevant $q$ - $\tau$-plane. Note that the symmetry relation $F_{a b}(\mathbf{q}, \tau)=F_{a b}(\mathbf{q}, \beta-\tau)$ holds [see also Fig. 3 below], as a consequence of the imaginary-time translation invariance in thermodynamic equilibrium or, equivalently, from the detailed balance relation $S_{a b}(\mathbf{q}, \omega)=e^{-\beta \omega} S_{a b}(\mathbf{q},-\omega)$ for the DSF $[53,108]$. Thus, we can restrict ourselves to the discussion of the interval $\tau \in[0, \beta / 2]$. The partial electron-electron ITCF $F_{e e}(\mathbf{q}, \tau)$ shown in the left panel exhibits a rich structure that is the combined result of multiple physical effects. The $\tau=0$ limit corresponds to the static structure factor $F_{a b}(\mathbf{q}, 0)=S_{a b}(\mathbf{q})$, with $S_{e e}(\mathbf{q})$ approaching unity for large wavenumbers, and a finite value for $q \rightarrow 0$ [88]. In addition, $F_{e e}(\mathbf{q}, \tau)$ exhibits an increasingly pronounced decay with $\tau$ for large $q$. From a physical perspective, this is due to the Gaussian imaginarytime diffusion process that governs the particle path in the path-integral picture $[108,109]$. In essence, the electrons are quantum delocalized, with the extension of their imaginary-time paths being proportional to the thermal wavelength as it has been explained in the discussion of Fig. 1 above. With increasing $q$, one effectively measures correlations on increasingly small length scales $\lambda=2 \pi / q$. For small $\lambda$, any correlations completely decay along the imaginary-time diffusion, and $F_{e e}(\mathbf{q}, \beta / 2)$ goes to zero. From a mathematical perspective, this increasing $\tau$-decay also follows from the f-sum rule, which states that $[19,108,111,143]$

$$
\begin{equation*}
\left.\frac{\partial}{\partial \tau} F_{a b}(\mathbf{q}, \tau)\right|_{\tau=0}=-\delta_{a b} \frac{q^{2}}{2 m_{a}} \tag{24}
\end{equation*}
$$

These different trends can also be seen in Fig. 3, where we show the $\tau$-dependence of $F_{e e}(\mathbf{q}, \tau)$ for $q=1.53 \AA^{-1}$ (top) and $q=7.65 \AA^{-1}$ (bottom) as the solid red curves. Additional insights come from a comparison with the corresponding UEG results at the same conditions, i.e., the double-dashed yellow curves. For the smaller $q$-value, the DSF of the UEG is dominated by a single broadened plasmon peak in the vicinity of the plasma frequency [144], leading to a moderate decay with $\tau$. For full hydrogen, we find a very similar $\tau$-dependence, but the entire curve appears to be shifted by a constant off-set; this is a direct signal of the elastic feature [Eq. (23)] in $S_{e e}(\mathbf{q}, \omega)$ that originates from the electronic localization around the protons. For $q=7.65 \AA^{-1}$, on the other hand, the UEG and
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-08.jpg?height=506&width=1828&top_left_y=280&top_left_x=148)

FIG. 2. Ab initio PIMC results for the partial imaginary-time density-density correlation functions of warm dense hydrogen for $N=32$ hydrogen atoms at the electronic Fermi temperature $\Theta=1$ and a metallic density $r_{s}=2$ in the $\tau$ - $q$-plane: electronelectron ITCF $F_{e e}(\mathbf{q}, \tau)$ [left], proton-proton ITCF $F_{p p}(\mathbf{q}, \tau)$ [center] and electron-proton ITCF $F_{e p}(\mathbf{q}, \tau)$ [right].

full hydrogen give very similar results for $F_{e e}(\mathbf{q}, \tau)$. In fact, they are indistinguishable around $\tau=0$, as it holds $S_{e e}(\mathbf{q})=1$ in the large- $q$ limit, and both curves exactly fulfill Eq. (24). Interestingly, we observe a slightly reduced $\tau$-decay in $\mathrm{H}$ compared to the UEG model around $\tau=\beta / 2$, which has small though significant implications for $\chi_{e e}(\mathbf{q})$ as we shall see below.

Let us next consider the proton-proton ITCF $F_{p p}(\mathbf{q}, \tau)$ and the electron-proton ITCF $F_{e p}(\mathbf{q}, \tau)$ which are shown in the center and right panels of Fig. 2. First and foremost, both correlation functions appear to be independent of $\tau$ over the entire depicted $q$-range. For $F_{e p}(\mathbf{q}, \tau)$, this is indeed the case within the given MC error bars, see also the blue curves in Fig. 3. This is consistent with the f-sum rule [Eq. (24)], which states that $F_{e p}(\mathbf{q}, \tau)$ is constant at least in the first order of $\tau$. The proton-proton ITCF exhibits a richer behaviour at a magnified view, see the green curves in Fig. 3. Within our PIMC simulations, the protons are treated as delocalized quantum particles just as the electrons, although their thermal wavelength is smaller by a factor of $1 / m_{p} \approx 0.02$. This is reflected by the less extended paths along the imaginary-time diffusion process, cf. Fig. 1 and, consequently, by a strongly reduced $\tau$-decay of $F_{p p}(\mathbf{q}, \tau)$ compared to $F_{e e}(\mathbf{q}, \tau)$. This is also reflected by the mass in the denominator of the RHS. of Eq. (24). Remarkably, for large $q$, we can resolve a small, yet significant $\tau$-dependence of $F_{p p}(\mathbf{q}, \tau)$ that is of the order of $\sim 0.1 \%$, see the right inset of the bottom panel of Fig. 3. Overall, it is still clear that the behaviour of both $F_{p p}(\mathbf{q}, \tau)$ and $F_{e p}(\mathbf{q}, \tau)$ is essentially governed by the corresponding static structure factors $S_{p p}(\mathbf{q})$ and $S_{e p}(\mathbf{q})$, while the $\tau$-dependence is essentially negligible for the linear density response in this regime.

We now proceed to the central topic of the present work, which is the investigation of the partial static density response of warm dense hydrogen. Fig. 4 depicts the static density response function $\chi_{a b}(\mathbf{q})$ as a function of the wavenumber $q$. The red, green, and blue symbols have been computed from the ITCF through Eq. (9) for the electron-electron, proton-proton, and electron-proton response, respectively. The crosses and diamonds correspond to $N=14$ and $N=32$ atoms and no dependence on the system size can be resolved. Before comparing the data sets with the other depicted models and calculations, we shall first summarize their main trends. i) the electron-proton density response $\chi_{e p}(\mathbf{q})=\chi_{p e}(\mathbf{q})$ has the same negative sign as $\chi_{e e}(\mathbf{q})$ and $\chi_{p p}(\mathbf{q})$, since the unperturbed protons would follow the induced density of the perturbed electrons, and vice versa. ii) the electron-proton density response $\chi_{e p}(\mathbf{q})$ monotonically decays with $q$ as the electron-proton coupling vanishes in the single-particle limit; the same trend also leads to the well-known decay of the elastic feature in $S_{e e}(\mathbf{q}, \omega)$ quantified by the correspondingly vanishing Rayleigh weight $W_{\mathrm{R}}(\mathbf{q})$. iii) the electron-electron density response $\chi_{e e}(\mathbf{q})$ is relatively flat for $q \lesssim 2 \AA^{-1}$ and monotonically decays for larger $q$. Eq. (9) directly implies that this is a quantum delocalization effect. In practice, the static density response is proportional to the area under the corresponding ITCF. The latter vanishes increasingly fast with $\tau$ with increasing $q$ as discussed above, leading to the observed reduction in $\chi_{e e}(\mathbf{q})$. Heuristically, this can be understood as follows. While the static density response of a single (or noninteracting) classical particle is wavenumber independent, the response of a delocalized quantum particle gets reduced when its thermal wavelength is comparable to the perturbation wavelength. In fact, a quantum particle will stop reacting all together when its extension is much larger than the excited wavelength. iv) the proton-proton density response $\chi_{p p}(\mathbf{q})$ increases with $q$ and seemingly becomes constant for $q \gtrsim 6 \AA^{-1}$. The reduction in $\chi_{p p}(\mathbf{q})$ for small $q$ is a con-
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-09.jpg?height=1240&width=812&top_left_y=190&top_left_x=190)

FIG. 3. Ab initio PIMC results for partial hydrogen ITCFs at $r_{s}=2$ and $\Theta=1$ for $q=1.53 \AA^{-1}$ (or $q=0.84 q_{\mathrm{F}}$ ) [top] and $q=7.65 \AA^{-1}$ (or $q=4.21 q_{\mathrm{F}}$ ) [bottom]; solid red: $F_{e e}(\mathbf{q}, \tau)$, dashed green: $F_{p p}(\mathbf{q}, \tau)$, dotted blue: $F_{e p}(\mathbf{q}, \tau)$, doubledashed yellow: UEG model [108]. The shaded intervals correspond to $1 \sigma$ error bars. The insets in the right panel show magnified segments around $F_{e p}(\mathbf{q}, \tau)$ and $F_{p p}(\mathbf{q}, \tau)$.

sequence of the proton-proton coupling (and its interplay with electrons), whereas its behaviour for large $q$ comes from the heavier proton mass and the resulting strongly reduced quantum delocalization. For completeness, note that we can resolve small deviations from the classical limit of $\chi_{p p}^{\mathrm{cl}}(\mathbf{q})=-n_{p} \beta$ for large wavenumbers. We also note that the general property $\left|\chi_{A A}^{\mathrm{q}}(\mathbf{q})\right| \leq\left|\chi_{A A}^{\mathrm{cl}}(\mathbf{q})\right|$, valid for static linear response functions associated with any hermitian operator $\hat{A}$ [28], should be respected by all the species-resolved density responses.

Equally interesting to these observations about the density response of full two-component hydrogen is their comparison to other models and calculations. The grey squares in Fig. 4 have been obtained from the direct perturbation approach, i.e., from independent PIMC simulations of hydrogen where one component has been per-

![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-09.jpg?height=898&width=830&top_left_y=188&top_left_x=1100)

FIG. 4. Ab initio PIMC results for the partial static density responses of hydrogen at $r_{s}=2$ and $\Theta=1$. Red, green, blue symbols: $\chi_{e e}(\mathbf{q}), \chi_{p p}(\mathbf{q}), \chi_{e p}(\mathbf{q})$ for full hydrogen evaluated from the ITCF via Eq. (9); grey squares: $\chi_{e e}(\mathbf{q}), \chi_{p p}(\mathbf{q})$, $\chi_{e p}(\mathbf{q})$ for full hydrogen evaluated from the direct perturbation approach [Eq. (7)]; black circles: electronic density response of fixed ion snapshot [74]; solid yellow line: UEG [71]; dashed black: ideal density responses $\chi_{e e}^{(0)}(\mathbf{q})$ and $\chi_{p p}^{(0)}(\mathbf{q})$.

turbed by an external harmonic perturbation, cf. Eq. (4). This procedure is further illustrated in Fig. 5, where we show the induced density $\rho_{a}(\mathbf{q})$ as a function of the perturbation amplitude. More specifically, the red crosses in the top panel correspond to the electronic density of full two-component hydrogen induced by the electronic perturbation amplitude $A_{e}$ (with $A_{p}=0$ ) for a wavenumber of $q=1.53 \AA^{-1}$. In the limit of $A_{e} \rightarrow 0, \rho_{e}(\mathbf{q}) / A_{e}$ attains a finite value that is given by the static linear density response function $\chi_{e e}(\mathbf{q})$; the latter is shown as the horizontal dashed blue line, as computed from the ITCF $F_{e e}(\mathbf{q}, \tau)$ via Eq. (9), and it is in excellent agreement with the red diamonds in the limit of small $A_{e}$. The solid green lines show cubic fits via Eq. (7). Evidently, the latter nicely reproduce the $A_{e}$ dependence of the PIMC data for moderate perturbations, and the linear coefficient then corresponds to the linear density response function; taking into account the deviations between data and fits for $A \gtrsim 10 \mathrm{eV}$ is possible by including higher-order terms in Eq. (7) [120], which will be pursued in future works. The red crosses have been obtained from the same set of simulations (i.e., $A_{e}>0$ and $A_{p}=0$ ) and depict the corresponding induced density of the protons that is described by $\chi_{e p}(\mathbf{q})$. The unperturbed protons
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-10.jpg?height=1300&width=832&top_left_y=190&top_left_x=186)

FIG. 5. Partial induced density for $q=1.53 \AA^{-1}$ as function of the perturbation strength $A$ [cf. Eq. (4)] at $r_{s}=2$ and $\Theta=1$ for $N=14 \mathrm{H}$ atoms. The dashed blue lines show the linearresponse limit computed from the ITCF via Eq. (9), and the solid green lines the cubic polynomial fits via Eq. (7). Top: the induced electronic density $\rho_{e}$ as function of $A_{e}$ (red stars, $A_{p}=0$ ) and $A_{p}$ (yellow crosses, $A_{e}=0$ ); the induced proton density $\rho_{p}$ as function of $A_{p}$ (yellow squares, $A_{e}=0$ ) and $A_{e}$ (red crosses, $A_{p}=0$ ). Bottom: the induced electronic density $\rho_{e}$ as function of the electronic perturbation strength $A_{e}$ for full hydrogen (red diamonds, $A_{p}=0$ ), a fixed proton snapshot [74] (black crosses), and the UEG [129] (yellow squares). Additional results are shown in Appendix B.

thus do indeed follow the perturbed electrons, although with a somewhat reduced magnitude. This can be discerned particularly well in Fig. 6, where we show the density in real space along the direction of the perturbation for a comparably small electronic perturbation amplitude of $A_{e}=1.36 \mathrm{eV}$. The solid red line shows the PIMC results for the electronic density and the dotted blue curve the linear-response theory estimate [129]

$$
\begin{equation*}
n_{e}(\mathbf{r})=n_{0}+2 A_{e} \cos (\mathbf{q} \cdot \mathbf{r}) \chi_{e e}(\mathbf{q}) \tag{25}
\end{equation*}
$$

using the ITCF based result for $\chi_{e e}(\mathbf{q})$. Both curves are

![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-10.jpg?height=691&width=810&top_left_y=191&top_left_x=1102)

FIG. 6. The induced density change $\Delta n(z) / n_{0}$ for an electronic perturbation amplitude of $A_{e}=0.05 \mathrm{Ha}\left(A_{e}=1.36 \mathrm{eV}\right.$ ) [cf. Eq. (4)] for $r_{s}=2$ and $\Theta=1$. Solid red: electron density $n_{e}(z)$ for full hydrogen (with $A_{p}=0$ ); dotted blue: corresponding linear-response prediction [Eq. (25)]; solid green: proton density $n_{p}(z)$ for full hydrogen (with $A_{p}=0$ ); solid yellow: electron density $n_{e}(z)$ for the UEG model [129]; dashed black: electron density $n_{e}(z)$ of a fixed proton snapshot [74].

in excellent agreement, which implies that LRT is accurate in this regime. The green curve shows the proton density from the same calculation. It exhibits the same cosinusoidal form, but with a reduced magnitude. Let us postpone a discussion of the other curves in Fig. 6 and return to the top panel of Fig. 5. The yellow squares show results for the induced proton density $\rho_{p}$ that have been obtained from a second, independent set of PIMC calculations with a finite proton perturbation amplitude $A_{p}>0$, but unperturbed electrons, $A_{e}=0$. We find excellent agreement with the ITCF based result for the LRT limit of $\chi_{p p}(\mathbf{q})$, and the cubic fit agrees well with the data. The protons react more strongly to an external static perturbation compared to the electrons, see the discussion of Fig. 4 above. Finally, the yellow crosses show the induced electron density $\rho_{e}$ from the same calculation. We recover the expected symmetry $\chi_{e p}(\mathbf{q})=\chi_{p e}(\mathbf{q})$ within the linear response limit. Interestingly, this breaks down for larger perturbation amplitudes, where the protons again react more strongly than the electrons. This nonlinear effect deserves further explanation, and will be investigated in detail in a dedicated future work.

In the appendix B, we show more results for the direct perturbation approach for different wavenumbers, which have been employed to obtain the linear density response functions that are shown as the empty squares in Fig. 4. We find perfect agreement with the ITCF based data sets everywhere.

Let us next consider the solid yellow curve in Fig. 4, which shows the density response of the UEG model [71]

![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-11.jpg?height=903&width=829&top_left_y=180&top_left_x=190)

FIG. 7. Ab initio PIMC results for the partial local field factors $\theta_{a b}(\mathbf{q})$ for $r_{s}=2$ and $\Theta=1$. Red, green, and blue: $\theta_{e e}(\mathbf{q})$, $\theta_{e p}(\mathbf{q})$, and $\theta_{p p}(\mathbf{q})$ of full hydrogen [Eqs. (14-16)]; yellow: electron-electron local field factor of the UEG model [71].

at the same conditions. We find good (though not perfect) agreement with the electronic density response of full hydrogen for $q \gtrsim 4 \AA^{-1}$. In stark contrast, there appear substantial differences between the two data sets for $\chi_{e e}(\mathbf{q})$ in the limit of small $q$, where the density response of the UEG vanishes due to perfect screening [145]. This is not the case for hydrogen due to the electron-proton coupling, which can be understood intuitively in two different ways. From the perspective of density response theory, the fact that we study the response in the static limit of $\omega \rightarrow 0$ implies that the protons have sufficient time to follow the perturbed electrons, thereby effectively screening the Coulomb interaction between the latter. Ionic mobility breaks down the perfect screening relation of the UEG, and allows the electrons to react even to perturbations on very large length scales (i.e., $q \rightarrow 0$ ), directly leading to the nonvanishing value of $\chi_{e e}(\mathbf{q})$ for large wavelengths. Additional insight comes from the relation of $\chi_{e e}(\mathbf{q})$ as the inverse frequency moment of the dynamic structure factor $S_{e e}(\mathbf{q}, \omega)$, cf. Eq. (19) in Sec. II D above. For the UEG, $S_{e e}(\mathbf{q}, \omega)$ simply consists of a sharp plasmon excitation around the (finite) plasma frequency for small $q$, and the weight of the plasmon vanishes quadratically in this regime [28]. For full twocomponent hydrogen, on the other hand, $S_{e e}(\mathbf{q}, \omega)$ contains additional contributions i) from bound-free transitions and ii) from the quasi-elastic feature that is usually modelled as a sum of an atomic form factor and a screening cloud of free electrons [137]. The latter feature increases with small $q$ and, being located at very small frequencies, strongly manifests in the inverse moment of $S_{e e}(\mathbf{q}, \omega)$; the static electron-electron density response function is thus highly sensitive to electronic localization around the protons. We note that this has potentially important implications for the interpretation of XRTS experiments with WDM, since $\chi_{e e}(\mathbf{q})$ can be directly inferred from the measured intensity (cf. Sec. IID) and, from a theoretical perspective, it does not require dynamic simulations. We thus suggest that, after having inferred the temperature from the model-free ITCF thermometry method introduced in Refs. [53, 110], one might calculate $\chi_{e e}(\mathbf{q})$ for a given wavenumber over a relevant interval of densities to infer the latter from the XRTS measurement. Such a strategy would completely circumvent the unphysical decomposition into bound and free electrons, while at the same time being very sensitive to a related, but well-defined concept: electronic localization around the ions. For light elements such as $H$ or Be [87], this might even be accomplished on the basis of quasi-exact PIMC results, offering a pathway for, in principle, approximation-free WDM diagnostics. At the same time, we point out that the static density response might also be estimated with reasonable accuracy from computationally less demanding methods such as DFT or restricted PIMC (using the direct perturbation approach) since no dynamic information is required. A dedicated exploration of this idea thus constitutes an important route for future research. Let us conclude our comparison between the electronic density response of full two-component hydrogen and the UEG by inspecting the corresponding density profile in Fig. 6, which is shown as the solid yellow curve. As it is expected, the UEG reacts less strongly to an equal external perturbation. Note that the nearly perfect agreement with the green curve is purely coincidental and is a consequence of the intersection of the yellow and blue data in Fig. 8 for the considered wavenumber.

Finally, the solid black dots in Fig. 4 have been adopted from Böhme et al. [74] and show $\chi_{e e}(\mathbf{q})$ of a non-uniform electron gas in the external potential of a fixed proton configuration. Evidently, keeping the protons fixed has a dramatic impact on the electronic density response. Those electrons that are located around a proton are substantially less likely to react to the external harmonic perturbation than the electrons in a free electron gas [146], leading to an overall reduction of the density response for small to intermediate $q$. In particular, this snapshot based separation of the electrons and protons completely misses the correct signal of the electronic localization around the protons that has been discussed in the previous paragraph. While being ideally suited to benchmark DFT simulations, and potentially to provide input to the latter, the physics content of these results is thus quite incomplete.

Let us conclude our analysis of warm dense hydrogen at $r_{s}=2$ by considering the various local field factors $\theta_{a b}(\mathbf{q})$,
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-12.jpg?height=502&width=1830&top_left_y=283&top_left_x=146)

FIG. 8. Ab initio PIMC results for the partial imaginary-time density-density correlation functions of warm dense hydrogen for $N=32$ hydrogen atoms at the electronic Fermi temperature $\Theta=1$ and a solid density $r_{s}=3.23$ in the $\tau$ - $q$-plane: electronelectron ITCF $F_{e e}(\mathbf{q}, \tau)$ left], proton-proton ITCF $F_{p p}(\mathbf{q}, \tau)$ [center] and electron-proton ITCF $F_{e p}(\mathbf{q}, \tau)$ [right].

cf. Eqs. (14)-(16), that are shown in Fig. 7. The solid yellow line corresponds to the machine learning (ML) representation of the UEG [71] and has been included as a reference. The solid red line shows the electron-electron local field factor $\theta_{e e}(\mathbf{q})$. It attains the same limit of -1 for $q \rightarrow 0$, but substantially deviates from the UEG result for all finite wavenumbers. The straightforward application of UEG models for the description of real hydrogen is thus questionable in this regime. This discrepancy becomes even more pronounced for solid state density, but vanishes for $r_{s}=1$, cf. Secs. III B and III C below. The green curve shows the proton-proton local field factor $\theta_{p p}(\mathbf{q})$. Interestingly, it is basically indistinguishable from $\theta_{e e}(\mathbf{q})$ for $q \lesssim 4 \AA^{-1}$, but diverges from the latter for large wavenumbers. Finally, the blue curve shows the electron-proton local field factor $\theta_{e p}(\mathbf{q})$. Unsurprisingly, it is larger in magnitude than the other data sets for small to moderate wavenumbers, which reflects the importance of electron-proton coupling effects.

## B. Solid density hydrogen

As a second example, we investigate hydrogen at $\Theta=1$ and $r_{s}=3.23$, i.e., the density of solid state hydrogen. From a physical perspective, the low density is expected to lead to an increased impact of both electron-electron and electron-proton coupling effects [74, 147, 148], making these conditions a very challenging test case for simulations. We note that, in contrast to the UEG, the lower density is also more challenging for our PIMC setup. This is a consequence of the incipient formation of $\mathrm{H}^{-}$ions and molecules [88], leading to a more substantial degree of quantum degeneracy and, consequently, a more severe sign problem. Indeed, we find an average sign of $S \approx 0.05$ for $N=14$ hydrogen atoms, causing a factor of $1 / S^{2}=400$ in the required compute time. In addition, low-density hydrogen is expected to exhibit interesting physical effects, such as the roton-type feature [149] in the dynamic structure factor $S_{e e}(\mathbf{q}, \omega)$ for intermediate wavenumbers [150]. Intriguingly, such conditions can be probed with XRTS measurements of optically pumped hydrogen jets [91, 92], which makes our results directly relevant for upcoming experiments.

In Fig. 8, we show our new PIMC results for the different ITCFs $F_{a b}(\mathbf{q}, \tau)$ in the relevant $\tau$ - $q$-plane. Overall, we find the same qualitative trends as with $r_{s}=2$ investigated above, but with two main differences: i) the electron-proton ITCF $F_{e p}(\mathbf{q}, \tau)$ attains larger values on the depicted $q$-grid, indicating a higher degree of coupling between the two species. ii) both $F_{e e}(\mathbf{q}, 0)$ and $F_{p p}(\mathbf{q}, 0)$ exhibit a reduced decay for small $q$, for the same reason.

Additional insight comes from Fig. 9, where we show the various ITCFs for $q=0.95 \AA^{-1}$ (top) and $q=4.73 \AA^{-1}$ (bottom) along the $\tau$-axis. In the top panel, the main difference from the $r_{s}=2$ case is the larger offset between the results for $F_{e e}(\mathbf{q}, \tau)$ from the UEG and full twocomponent hydrogen; it is a direct consequence of the increased electronic localization around the protons and the correspondingly increased Rayleigh weight $W_{R}(\mathbf{q})$. For the larger $q$-value, we again find that no dependence of $F_{e p}(\mathbf{q}, \tau)$ can be resolved within the given confidence interval (shaded blue area). In contrast, we can clearly resolve protonic quantum effects, see the right inset. An additional interesting observation comes from a comparison of the solid red and double-dashed yellow curves corresponding to $F_{e e}(\mathbf{q}, \tau)$ for hydrogen and the UEG. In the limit of $\tau \rightarrow 0$, the two curves are in perfect agreement; this is a consequence of the fact that $S_{e e}(\mathbf{q}) \approx 1$ and the fsum rule [Eq. (24)] yielding the same slope for both data sets. For larger $\tau$, on the other hand, we observe substantial differences between hydrogen and the UEG. Specif-
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-13.jpg?height=1248&width=832&top_left_y=186&top_left_x=186)

FIG. 9. Ab initio PIMC results for partial hydrogen ITCFs at $r_{s}=3.23$ and $\Theta=1$ for $q=0.95 \AA^{-1}$ (or $q=0.84 q_{\mathrm{F}}$ ) [top] and $q=4.73 \AA^{-1}$ (or $q=4.21 q_{\mathrm{F}}$ ) [bottom]; solid red: $F_{e e}(\mathbf{q}, \tau)$, dashed green: $F_{p p}(\mathbf{q}, \tau)$, dotted blue: $F_{e p}(\mathbf{q}, \tau)$, double-dashed yellow: UEG model [108]. The shaded intervals correspond to $1 \sigma$ error bars. The insets in the right panel show magnified segments around $F_{e p}(\mathbf{q}, \tau)$ and $F_{p p}(\mathbf{q}, \tau)$.

ically, we find a reduced $\tau$-decay for the former system compared to the latter, which cannot simply be explained by a constant shift due to the Rayleigh weight. From the perspective of our PIMC simulations, this clearly indicates that electron-electron correlations are stabilized along the imaginary-time diffusion process by the presence of the protons. Equivalently, we can attribute this finding to a shift of spectral weight in $S_{\text {inel }}(\mathbf{q}, \omega)$ to lower frequencies (see also Ref. [108] for a more detailed discussion), indicating a nontrivial structure of the full DSF. The presented data for $F_{e e}(\mathbf{q}, \tau)$ thus constitute rigorous benchmarks for models (e.g., the Chihara decomposition) and simulations (e.g., LR-TDDFT), and a dedicated future comparative analysis will give important insights into the validity range of different methods.

In Fig. 10, we show the corresponding species-resolved static density response functions $\chi_{a b}(\mathbf{q})$, and again re-

![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-13.jpg?height=876&width=808&top_left_y=191&top_left_x=1103)

FIG. 10. Ab initio PIMC results for the partial static density responses of hydrogen at $r_{s}=3.23$ and $\Theta=1$. Red, green, blue symbols: $\chi_{e e}(\mathbf{q}), \chi_{p p}(\mathbf{q}), \chi_{e p}(\mathbf{q})$ for full hydrogen evaluated from the ITCF via Eq. (9); solid yellow line: UEG model [71]; dashed black: ideal density responses $\chi_{e e}^{(0)}(\mathbf{q})$ and $\chi_{p p}^{(0)}(\mathbf{q})$.

![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-13.jpg?height=889&width=808&top_left_y=1388&top_left_x=1103)

FIG. 11. Ab initio PIMC results for the partial local field factors $\theta_{a b}(\mathbf{q})$ for $r_{s}=3.23$ and $\Theta=1$. Red, green, and blue: $\theta_{e e}(\mathbf{q}), \theta_{e p}(\mathbf{q})$, and $\theta_{p p}(\mathbf{q})$ of full hydrogen [Eqs. (14-16)]; yellow: local field factor of the UEG model [71].
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-14.jpg?height=504&width=1828&top_left_y=282&top_left_x=149)

FIG. 12. Ab initio PIMC results for the partial imaginary-time density-density correlation functions of warm dense hydrogen for $N=32$ hydrogen atoms at the electronic Fermi temperature $\Theta=1$ and a compressed density $r_{s}=1$ in the $\tau$ - $q$-plane: electron-electron ITCF $F_{e e}(\mathbf{q}, \tau)$ [left], proton-proton ITCF $F_{p p}(\mathbf{q}, \tau)$ [center] and electron-proton ITCF $F_{e p}(\mathbf{q}, \tau)$ [right].

strict ourselves to a discussion of the main differences from the $r_{s}=2$ case. i) $\chi_{e e}(\mathbf{q})$ monotonically increases with decreasing $q$ in over the entire depicted $q$-range, and the electronic localization around the protons predominantly shapes its behaviour for small $q$. ii) in contrast to the UEG, $\chi_{e e}(\mathbf{q})$ does not converge towards the ideal density response $\chi_{e e}^{(0)}(\mathbf{q})$ for large $q$, which is a direct consequence of the reduced $\tau$-decay of $F_{e e}(\mathbf{q}, \tau)$ depicted in the bottom panel of Fig. 8 above. iii) $\chi_{e p}(\mathbf{q})$ is substantially larger, with respect to $\chi_{e e}(\mathbf{q})$ and $\chi_{p p}(\mathbf{q})$, for $r_{s}=3.23$ compared to $r_{s}=2$. iv) $\chi_{p p}(\mathbf{q})$ exhibits a reduced decay for small $q$ compared to $r_{s}=2$; this is due to the electronic screening of the proton-proton interaction, making the proton response more ideal.

Finally, we show the partial local field factors $\theta_{a b}(\mathbf{q})$ in Fig. 11. While the electron-electron local field factor of full two-component hydrogen (solid red) attains the same limit as the UEG model [71] (yellow) for $q \rightarrow 0$, it exhibits the opposite trend for intermediate $q$, followed by a steep increase for shorter wavelengths. This is consistent with previous findings by Böhme et al. [74] for an electron gas in a fixed external proton potential at similar conditions $\left(r_{s}=4\right)$ and clearly indicates the breakdown of UEG models when electrons are strongly localized. The proton-proton local field factor is relatively featureless over the entire $q$-range, which might be due to the aforementioned electronic screening of the proton-proton interaction. Similar to $r_{s}=2, \theta_{e p}(\mathbf{q})$ constitutes the largest local field factor for relevant wave numbers.

## C. Strongly compressed hydrogen

As the final example, we investigate compressed hydrogen at $\Theta=1$ and $r_{s}=1$. The corresponding PIMC results for the species-resolved ITCFs are shown in Fig. 12 in the $\tau$-q-plane and qualitatively closely resemble the case of $r_{s}=2$ shown in Fig. 3 above. The main difference is the reduced magnitude of $F_{e p}(\mathbf{q}, \tau)$, indicating a substantially weaker localization of the electrons around the protons, as it is expected. In Fig. 13, we show the ITCFs along the $\tau$-direction for $q=3.06 \AA^{-1}$ and $q=15.3 \AA^{-1}$, i.e, for the same values of $q / q_{\mathrm{F}}$ as in Figs. 3 and 9. We find very similar behaviour as for $r_{s}=2$ with a reduced electron-proton coupling. In fact, no deviations can be resolved between $F_{e e}(\mathbf{q}, \tau)$ for the UEG model and full two-component hydrogen in the bottom panel. In other words, compressed hydrogen closely resembles a free electron gas.

This observation is further substantiated in Fig. 14, where we show the corresponding partial static density response functions $\chi_{a b}(\mathbf{q})$. Evidently, $\chi_{e e}(\mathbf{q})$ is very similar to the UEG, and converges towards the latter for $q \gtrsim 5 \AA^{-1}$. Nevertheless, we can still clearly detect the signature of electronic localization around the protons as a somewhat increased response for smaller $q$. In fact, the convergence of the electronic density response of hydrogen towards the UEG model for high densities can be seen most clearly in Fig. 15 where we show the partial local field factors $\theta_{a b}(\mathbf{q})$. In stark contrast to $r_{s}=2$ and mainly to $r_{s}=3.23$, the UEG model is in very good agreement with the true $\theta_{e e}(\mathbf{q})$ of hydrogen at $r_{s}=1$. This has important implications for laser fusion applications and clearly indicates that UEG based models such as the adiabatic local density approximation are appropriate over substantial parts of the ICF compression path. Finally, we note that the electron-proton local field factor $\theta_{e p}(\mathbf{q})$ has the same magnitude as $\theta_{e e}(\mathbf{q})$ in this case, whereas $\theta_{p p}(\mathbf{q})$ is somewhat smaller.
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-15.jpg?height=1232&width=806&top_left_y=194&top_left_x=190)

FIG. 13. Ab initio PIMC results for partial hydrogen ITCFs at $r_{s}=1$ and $\Theta=1$ for $q=3.06 \AA^{-1}$ (or $q=0.84 q_{\mathrm{F}}$ ) [top] and $q=15.3 \AA^{-1}$ (or $q=4.21 q_{\mathrm{F}}$ ) [bottom]; solid red: $F_{e e}(\mathbf{q}, \tau)$, dashed green: $F_{p p}(\mathbf{q}, \tau)$, dotted blue: $F_{e p}(\mathbf{q}, \tau)$, double-dashed yellow: UEG model [108]. The shaded intervals correspond to $1 \sigma$ error bars. The insets in the right panel show magnified segments around $F_{e p}(\mathbf{q}, \tau)$ and $F_{p p}(\mathbf{q}, \tau)$.

## IV. SUMMARY AND DISCUSSION

In this work, we have presented the first $a b$ initio results for the partial density response functions of warm dense hydrogen. This has been achieved on the basis of direct PIMC simulations that are computationally very expensive owing to the fermion sign problem, but exact within the given Monte Carlo error bars. Moreover, we have employed the recently introduced $\xi$-extrapolation technique [84-88] to access larger system sizes; no finitesize effects have been detected for the wavenumberresolved properties, in agreement with previous results for the UEG model at the same conditions [89]. A particular advantage of the direct PIMC method is that it allows us to estimate all ITCFs $F_{a b}(\mathbf{q}, \tau)$. First and foremost, this gives us direct access to the full wavenumber

![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-15.jpg?height=892&width=826&top_left_y=194&top_left_x=1105)

FIG. 14. Ab initio PIMC results for the partial static density responses of hydrogen at $r_{s}=1$ and $\Theta=1$. Red, green, blue symbols: $\chi_{e e}(\mathbf{q}), \chi_{p p}(\mathbf{q}), \chi_{e p}(\mathbf{q})$ for full hydrogen evaluated from the ITCF via Eq. (9); solid yellow line: UEG model [71]; dashed black: ideal density responses $\chi_{e e}^{(0)}(\mathbf{q})$ and $\chi_{p p}^{(0)}(\mathbf{q})$.

dependence of the static density response $\chi_{a b}(\mathbf{q})$ and, consequently, the local field factor $\theta_{a b}(\mathbf{q})$ from a single simulation of the unperturbed system. As an additional crosscheck, we have also carried out extensive simulations of hydrogen where either the electrons or the protons are subject to an external monochromatic perturbation. We find perfect agreement between the direct perturbation approach and the ITCF-based method in the linear response regime, as it is expected.

In addition to the anticipated impact on future investigations of WDM that is outlined below, the presented study is highly interesting in its own right and has given new insights into the complex interplay of the electrons and protons in different regimes. We repeat that both $F_{e e}(\mathbf{q}, \tau)$ and $\chi_{e e}(\mathbf{q})$ can be obtained from XRTS measurements, and our results thus constitute unambiguous predictions for experiments with ICF plasmas and hydrogen jets. In particular, we have shown that $\chi_{e e}(\mathbf{q})$ is highly sensitive to the electronic localization around the protons. This effect is particularly pronounced for small wavenumbers, which can be probed in forward scattering geometries [51], and it is directly related to the important concept of effective ionization. At the same time, we stress that the latter is, strictly speaking, ill-defined and ambiguous, whereas the reported impact of electronic localization on $\chi_{e e}(\mathbf{q})$ constitutes a well-defined physical observable both in experiments and simulations. In terms

![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-16.jpg?height=892&width=807&top_left_y=194&top_left_x=190)

FIG. 15. Ab initio PIMC results for the partial local field factors $\theta_{a b}(\mathbf{q})$ for $r_{s}=1$ and $\Theta=1$. Red, green, and blue: $\theta_{e e}(\mathbf{q}), \theta_{e p}(\mathbf{q})$, and $\theta_{p p}(\mathbf{q})$ of full hydrogen [Eqs. (14-16)]; yellow: local field factor of the UEG model [71].

of physical parameters, we have found that the electrons in hydrogen behave very differently from the UEG model for $r_{s}=2$ and even more so for $r_{s}=3.23$, while the UEG model is appropriate for strongly compressed hydrogen at $r_{s}=1$. Finally, we have reported, to the best of our knowledge, the first reliable results for the local field factor $\theta_{a b}(\mathbf{q})$ that is directly related to the XC-kernel from LR-TDDFT calculations.

We are convinced that our study opens up a wealth of opportunities for impactful future work, and helps to further lift WDM theory onto the level of true predictive capability. In the following, we give a non-exhaustive list of particularly promising projects.

i) The electron-electron density response function $\chi_{e e}(\mathbf{q})$ is ideally suited for the model-free interpretation of XRTS measurements of WDM. Specifically, we propose to first infer the temperature from the exact ITCFbased thermometry method introduced in Refs. [53, 110] and to subsequently carry out a set of PIMC simulations for $\chi_{e e}(\mathbf{q})$ over a relevant set of densities. Matching the PIMC result to the experimental result for $\chi_{e e}(\mathbf{q})$, the inverse moment of $S_{e e}(\mathbf{q}, \omega)$, then gives one model-free access to the density. This PIMC based interpretation framework of XRTS experiments will give new insights into the equation-of-state of WDM, with important implications for astrophysical models and laser fusion applications. Additionally, we note that computing $\chi_{e e}(\mathbf{q})$ does not require dynamical simulations or dynamic XC- kernels, and, therefore, might be suitable for approximate methods such as DFT.

ii) The species-resolved local field factors $\theta_{a b}(\mathbf{q})$, and in particular the electron-electron XC-kernel, constitute key input for a gamut of applications, including the estimation of thermal and electrical conductivities [61], the construction of effective potentials [66, 151, 152], and the estimation of the ionization potential depression [65]. Two particularly enticing applications concern the construction of nonlocal XC-functionals for DFT simulations based on the adiabatic connection formula and the fluctuation-dissipation theorem [68], and LR-TDDFT simulations within the adiabatic approximation [39, 115]. In fact, previous studies of the UEG [72, 73, 124, 144] have reported that the utilization of a static XC-kernel is capable of giving highly accurate results for $S_{e e}(\mathbf{q}, \omega)$ over substantial parts of the WDM regime. Extending these considerations for hydrogen and beyond are promising routes that will be explored in dedicated future works.

iii) Our quasi-exact PIMC results constitute a rigorous benchmark for computationally less expensive though approximate simulation methods, most importantly thermal DFT. Therefore, a dedicated comparative investigation for a real WDM system will give invaluable new insights into the range of applicability of available XCfunctionals, and guide the development of new thermal functionals that are explicitly designed for application in the WDM regime [43, 46-49]. In addition, the presented results for both $F_{e e}(\mathbf{q}, \tau)$ and $\chi_{e e}(\mathbf{q})$ can be used to benchmark dynamic methods such as LRTDDFT [38, 39], real-time TDDFT [153], or indeed the popular but uncontrolled Chihara models [51, 54, 58, 93].

iv) A less straightforward, though highly rewarding endeavour is given by the so-called analytic continuation of $F_{e e}(\mathbf{q}, \tau)$, i.e., the numerical inversion of Eq. (20) to solve for the dynamic structure factor $S_{e e}(\mathbf{q}, \omega)$. While such an inverse Laplace transform constitutes a notoriously difficult and, in fact, ill-posed problem [154], this issue has been circumvented recently for the warm dense UEG based on the stochastic sampling of the dynamic local field factor with rigorous constraints imposed on the trial solutions [124, 144]. Finding similar constraints for warm dense hydrogen would open up the way for the first exact results for the DSF of real WDM systems as well as for related properties such as the dynamic dielectric function, conductivity, and dynamic density response $[113,114]$.

v) Finally, we note that current direct PIMC capabilities allow for highly accurate simulations of elements up to Be $[87,88]$. It will be very interesting to see how the more complex behaviour of such systems (e.g., double occupation of the $\mathrm{K}$ shell of Be for $T=100 \mathrm{eV}[87,88]$ ) manifests in the different ITCFs and density response functions. Moreover, these considerations might be extended to material mixtures such as $\mathrm{LiH}$, giving rise to additional cross correlations that can be straightforwardly estimated by upcoming PIMC simulations.

## Appendix A: Fermion sign problem and $\xi$-extrapolation method

With the objective to verify the absence of finite-size effects in the partial density response functions $\chi_{a b}(\mathbf{q})$ reported in the main text, we have simulated $N=32$ hydrogen atoms using the $\xi$-extrapolation method that has been originally proposed by Xiong and Xiong [86], and further explored in Refs. [84, 85, 87, 88, 155]. Here the basic idea is to carry out a set of PIMC simulations with different values for the fictitious spin variable $\xi \in$ $[-1,1]$, and to extrapolate from the sign-problem free domain of $\xi \geq 0$ to the correct fermionic limit of $\xi=-1$ via the empirical quadratic relation

$$
\begin{equation*}
A(\xi)=a_{0}+a_{1} \xi+a_{2} \xi^{2} \tag{A1}
\end{equation*}
$$

In practice, the reliability of the $\xi$ - extrapolation method can be ensured by checking its validity for a rather moderate system size, where direct fermionic PIMC simulations are still feasible.

In Fig. 16, we show a corresponding analysis for the electronic density response function $\chi_{e e}(\mathbf{q})$ at $\Theta=1$ for all three values of the density considered in this work. Specifically, the green crosses show the exact fermionic PIMC results for $\xi=-1$, and the red circles have been extrapolated from the sign-problem free domain (shaded grey area) via Eq. (A1). We find perfect agreement between the direct PIMC result and $\xi$-extrapolated PIMC result everywhere.

## Appendix B: Direct perturbation results

In Fig. 17, we show additional results for the induced electron density [top] and proton density [bottom] as a function of the electronic perturbation amplitude $A_{e}$ (with $A_{p}=0$ ) for a variety of wavenumbers $q$. As it is expected, the induced density in the limit of $A_{e} \rightarrow 0$ always converges towards the LRT limit (dashed blue lines) that we compute from the ITCF via Eq. (9).

## ACKNOWLEDGMENTS

This work was partially supported by the Center for Advanced Systems Understanding (CASUS), financed by Germany's Federal Ministry of Education and Research (BMBF) and the Saxon state government out of the State budget approved by the Saxon State Parliament. This work has received funding from the European Research Council (ERC) under the European Union's Horizon 2022 research and innovation programme (Grant agreement No. 101076233, "PREXTREME"). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the Euro-
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-17.jpg?height=1876&width=830&top_left_y=187&top_left_x=1100)

FIG. 16. Application of the $\xi$-extrapolation method for the electron-electron density response $\chi_{e e}(\mathbf{q})$ of hydrogen at $\Theta=1$ and three different densities. Green crosses: direct fermionic PIMC results for $\xi=-1$; red circles: extrapolation from the sign-problem free domain of $\xi \in[0,1]$ (shaded grey area) via Eq. (A1).
![](https://cdn.mathpix.com/cropped/2024_06_04_a25f7988b6e5d7d90b3dg-18.jpg?height=1252&width=832&top_left_y=179&top_left_x=186)

FIG. 17. Partial induced electronic $\rho_{e}(q)$ [top] and proton densities $\rho_{p}(q)$ [bottom] as a function of the electronic perturbation amplitude $A_{e}$ [cf. Eq. (4)] at $r_{s}=2$ and $\Theta=1$ for $N=14$ hydrogen atoms. Dashed blue: linear response limit evaluated from the ITCF [Eq. (9)]; solid green: cubic fits via Eq. (7); the different symbols distinguish different perturbation wave numbers $q$. pean Union nor the granting authority can be held responsible for them. Computations were performed on a Bull Cluster at the Center for Information Services and High-Performance Computing (ZIH) at Technische Universität Dresden, at the Norddeutscher Verbund für Hoch- und Höchstleistungsrechnen (HLRN) under grant mvp00024, and on the HoreKa supercomputer funded by the Ministry of Science, Research and the Arts BadenWürttemberg and by the Federal Ministry of Education and Research.
[1] F. Graziani, M. P. Desjarlais, R. Redmer, and S. B. Trickey, eds., Frontiers and Challenges in Warm Dense Matter (Springer, International Publishing, 2014).

[2] R.P. Drake, High-Energy-Density Physics: Foundation of Inertial Fusion and Experimental Astrophysics, Graduate Texts in Physics (Springer International Publishing, 2018).

[3] R. Betti and O. A. Hurricane, "Inertial-confinement fusion with lasers," Nature Physics 12, 435-448 (2016).

[4] S. X. Hu, B. Militzer, V. N. Goncharov, and S. Skupsky, "First-principles equation-of-state table of deuterium for inertial confinement fusion applications," Phys. Rev. B 84, 224109 (2011).

[5] E. I. Moses, R. N. Boyd, B. A. Remington, C. J. Keane, and R. Al-Ayat, "The national ignition facility: Ushering in a new age for high energy density science,"
Physics of Plasmas 16, 041006 (2009)

[6] A. B. Zylstra, O. A. Hurricane, D. A. Callahan, A. L. Kritcher, J. E. Ralph, H. F. Robey, J. S. Ross, C. V. Young, K. L. Baker, D. T. Casey, T. Döppner, L. Divol, M. Hohenberger, S. Le Pape, A. Pak, P. K. Patel, R. Tommasini, S. J. Ali, P. A. Amendt, L. J. Atherton, B. Bachmann, D. Bailey, L. R. Benedetti, L. Berzak Hopkins, R. Betti, S. D. Bhandarkar, J. Biener, R. M. Bionta, N. W. Birge, E. J. Bond, D. K. Bradley, T. Braun, T. M. Briggs, M. W. Bruhn, P. M. Celliers, B. Chang, T. Chapman, H. Chen, C. Choate, A. R. Christopherson, D. S. Clark, J. W. Crippen, E. L. Dewald, T. R. Dittrich, M. J. Edwards, W. A. Farmer, J. E. Field, D. Fittinghoff, J. Frenje, J. Gaffney, M. Gatu Johnson, S. H. Glenzer, G. P. Grim, S. Haan, K. D. Hahn, G. N. Hall, B. A. Hammel, J. Harte, E. Har-
touni, J. E. Heebner, V. J. Hernandez, H. Herrmann, M. C. Herrmann, D. E. Hinkel, D. D. Ho, J. P. Holder, W. W. Hsing, H. Huang, K. D. Humbird, N. Izumi, L. C. Jarrott, J. Jeet, O. Jones, G. D. Kerbel, S. M. Kerr, S. F. Khan, J. Kilkenny, Y. Kim, H. Geppert Kleinrath, V. Geppert Kleinrath, C. Kong, J. M. Koning, J. J. Kroll, M. K. G. Kruse, B. Kustowski, O. L. Landen, S. Langer, D. Larson, N. C. Lemos, J. D. Lindl, T. Ma, M. J. MacDonald, B. J. MacGowan, A. J. Mackinnon, S. A. MacLaren, A. G. MacPhee, M. M. Marinak, D. A. Mariscal, E. V. Marley, L. Masse, K. Meaney, N. B. Meezan, P. A. Michel, M. Millot, J. L. Milovich, J. D. Moody, A. S. Moore, J. W. Morton, T. Murphy, K. Newman, J.-M. G. Di Nicola, A. Nikroo, R. Nora, M. V. Patel, L. J. Pelz, J. L. Peterson, Y. Ping, B. B. Pollock, M. Ratledge, N. G. Rice, H. Rinderknecht, M. Rosen, M. S. Rubery, J. D. Salmonson, J. Sater, S. Schiaffino, D. J. Schlossberg, M. B. Schneider, C. R. Schroeder, H. A. Scott, S. M. Sepke, K. Sequoia, M. W. Sherlock, S. Shin, V. A. Smalyuk, B. K. Spears, P. T. Springer, M. Stadermann, S. Stoupin, D. J. Strozzi, L. J. Suter, C. A. Thomas, R. P. J. Town, E. R. Tubman, P. L. Volegov, C. R. Weber, K. Widmann, C. Wild, C. H. Wilde, B. M. Van Wonterghem, D. T. Woods, B. N. Woodworth, M. Yamaguchi, S. T. Yang, and G. B. Zimmerman, "Burning plasma achieved in inertial fusion," Nature 601, 542-548 (2022).

[7] O. A. Hurricane, P. K. Patel, R. Betti, D. H. Froula, S. P. Regan, S. A. Slutz, M. R. Gomez, and M. A. Sweeney, "Physics principles of inertial confinement fusion and u.s. program overview," Rev. Mod. Phys. 95, 025005 (2023).

[8] Abu-Shawareb et al. (The Indirect Drive ICF Collaboration), "Achievement of target gain larger than unity in an inertial fusion experiment," Phys. Rev. Lett. 132, 065102 (2024).

[9] Dimitri Batani, Arnaud Colaïtis, Fabrizio Consoli, Colin N. Danson, Leonida Antonio Gizzi, Javier Honrubia, Thomas Kühl, Sebastien Le Pape, Jean-Luc Miquel, Jose Manuel Perlado, and et al., "Future for inertial-fusion energy in europe: a roadmap," High Power Laser Science and Engineering 11, e83 (2023).

[10] B. Militzer, W. B. Hubbard, J. Vorberger, I. Tamblyn, and S. A. Bonev, "A massive core in jupiter predicted from first-principles simulations," The Astrophysical Journal 688, L45-L48 (2008).

[11] Alessandra Benuzzi-Mounaix, Stéphane Mazevet, Alessandra Ravasio, Tommaso Vinci, Adrien Denoeud, Michel Koenig, Nourou Amadou, Erik Brambrink, Floriane Festa, Anna Levy, Marion Harmand, Stéphanie Brygoo, Gael Huser, Vanina Recoules, Johan Bouchet, Guillaume Morard, François Guyot, Thibaut de Resseguier, Kohei Myanishi, Norimasa Ozaki, Fabien Dorchies, Jerôme Gaudin, Pierre Marie Leguay, Olivier Peyrusse, Olivier Henry, Didier Raffestin, Sebastien Le Pape, Ray Smith, and Riccardo Musella, "Progress in warm dense matter study with applications to planetology," Phys. Scripta T161, 014060 (2014).

[12] Ravit Helled, Guglielmo Mazzola, and Ronald Redmer, "Understanding dense hydrogen at planetary conditions," Nature Reviews Physics 2, 562-574 (2020).

[13] A. Becker, W. Lorenzen, J. J. Fortney, N. Nettelmann, M. Schöttler, and R. Redmer, "Ab initio equations of state for hydrogen (h-reos.3) and helium (he-reos.3) and their implications for the interior of brown dwarfs," Astrophys. J. Suppl. Ser 215, 21 (2014).

[14] D. Saumon, W. B. Hubbard, G. Chabrier, and H. M. van Horn, "The role of the molecular-metallic transition of hydrogen in the evolution of jupiter, saturn, and brown dwarfs," Astrophys. J 391, 827-831 (1992).

[15] Nadine Nettelmann, Bastian Holst, André Kietzmann, Martin French, Ronald Redmer, and David Blaschke, "Ab initio equation of state data for hydrogen, helium, and water and the internal structure of jupiter," The Astrophysical Journal 683, 1217 (2008).

[16] N. Nettelmann, A. Becker, B. Holst, and R. Redmer, "Jupiter models with improved ab initio hydrogen equation of state (h-reos.2)," The Astrophysical Journal 750, 52 (2012).

[17] Burkhard Militzer, William B. Hubbard, Sean Wahl, Jonathan I. Lunine, Eli Galanti, Yohai Kaspi, Yamila Miguel, Tristan Guillot, Kimberly M. Moore, Marzia Parisi, John E. P. Connerney, Ravid Helled, Hao Cao, Christopher Mankovich, David J. Stevenson, Ryan S. Park, Mike Wong, Sushil K. Atreya, John Anderson, and Scott J. Bolton, "Juno spacecraft measurements of jupiter's gravity imply a dilute core," The Planetary Science Journal 3, 185 (2022).

[18] Carlo Pierleoni, Miguel A. Morales, Giovanni Rillo, Markus Holzmann, and David M. Ceperley, "Liquid-liquid phase transition in hydrogen by coupled electron-ion monte carlo simulations," Proceedings of the National Academy of Sciences 113, 4953-4957 (2016).

[19] Tobias Dornheim, Zhandos A. Moldabekov, Kushal Ramakrishna, Panagiotis Tolias, Andrew D. Baczewski, Dominik Kraus, Thomas R. Preston, David A. Chapman, Maximilian P. Böhme, Tilo Döppner, Frank Graziani, Michael Bonitz, Attila Cangi, and Jan Vorberger, "Electronic density response of warm dense matter," Physics of Plasmas 30, 032705 (2023).

[20] A. V. Filinov and M. Bonitz, "Equation of state of partially ionized hydrogen and deuterium plasma revisited," Phys. Rev. E 108, 055212 (2023).

[21] Guglielmo Mazzola, Seiji Yunoki, and Sandro Sorella, "Unexpectedly high pressure for molecular dissociation in liquid hydrogen by electronic simulation," Nature Communications 5, 3487 (2014).

[22] Bingqing Cheng, Guglielmo Mazzola, Chris J. Pickard, and Michele Ceriotti, "Evidence for supercritical behaviour of high-pressure liquid hydrogen," Nature 585, 217-220 (2020).

[23] Valentin V. Karasiev, Joshua Hinz, S. X. Hu, and S. B. Trickey, "On the liquid-liquid phase transition of dense hydrogen," Nature 600, E12-E14 (2021).

[24] M. D. Knudson, M. P. Desjarlais, A. Becker, R. W. Lemke, K. R. Cochrane, M. E. Savage, D. E. Bliss, T. R. Mattsson, and R. Redmer, "Direct observation of an abrupt insulator-to-metal transition in dense liquid deuterium," Science 348, 1455-1460 (2015).

[25] Ranga P. Dias and Isaac F. Silvera, "Observation of the wigner-huntington transition to metallic hydrogen," Science 355, 715-718 (2017).

[26] Peter M. Celliers, Marius Millot, Stephanie Brygoo, R. Stewart McWilliams, Dayne E. Fratanduono, J. Ryan Rygg, Alexander F. Goncharov, Paul Loubeyre, Jon H. Eggert, J. Luc Peterson, Nathan B. Meezan, Sebastien Le Pape, Gilbert W. Collins, Raymond Jeanloz, and Russell J. Hemley, "Insulator-metal transition in
dense fluid deuterium," Science 361, 677-682 (2018).

[27] M. Bonitz, T. Dornheim, Zh. A. Moldabekov, S. Zhang, P. Hamann, H. Kählert, A. Filinov, K. Ramakrishna, and J. Vorberger, "Ab initio simulation of warm dense matter," Physics of Plasmas 27, 042710 (2020).

[28] G. Giuliani and G. Vignale, Quantum Theory of the Electron Liquid (Cambridge University Press, Cambridge, 2008).

[29] Torben Ott, Hauke Thomsen, Jan Willem Abraham, Tobias Dornheim, and Michael Bonitz, "Recent progress in the theory and simulation of strongly correlated plasmas: phase transitions, transport, quantum, and magnetic field effects," The European Physical Journal D 72, 84 (2018).

[30] G. Stefanucci and R. van Leeuwen, Nonequilibrium Many-Body Theory of Quantum Systems: A Modern Introduction (Cambridge University Press, 2013).

[31] M. Bonitz, A. Filinov, V. O. Golubnychiy, Th. Bornath, and W. D. Kraeft, "First principle thermodynamic and dynamic simulations for dense quantum plasmas," Contributions to Plasma Physics 45, 450-458 (2005).

[32] N. David Mermin, "Thermal properties of the inhomogeneous electron gas," Phys. Rev. 137, A1441-A1443 (1965).

[33] Bastian Holst, Ronald Redmer, and Michael P. Desjarlais, "Thermophysical properties of warm dense hydrogen using quantum molecular dynamics simulations," Phys. Rev. B 77, 184201 (2008).

[34] Kushal Ramakrishna, Tobias Dornheim, and Jan Vorberger, "Influence of finite temperature exchangecorrelation effects in hydrogen," Phys. Rev. B 101, 195129 (2020).

[35] J-F. Danel, L. Kazandjian, and R. Piron, "Equation of state of carbon in the warm dense matter regime from density-functional theory molecular dynamics," Phys. Rev. E 98, 043204 (2018).

[36] A. J. White and L. A. Collins, "Fast and universal kohnsham density functional theory algorithm for warm dense matter to hot dense plasma," Phys. Rev. Lett. 125, 055002 (2020).

[37] Kushal Ramakrishna, Attila Cangi, Tobias Dornheim, Andrew Baczewski, and Jan Vorberger, "Firstprinciples modeling of plasmons in aluminum under ambient and extreme conditions," Phys. Rev. B 103, 125118 (2021).

[38] Maximilian Schörner, Mandy Bethkenhagen, Tilo Döppner, Dominik Kraus, Luke B. Fletcher, Siegfried H. Glenzer, and Ronald Redmer, "X-ray thomson scattering spectra from density functional theory molecular dynamics simulations based on a modified chihara formula," Phys. Rev. E 107, 065207 (2023).

[39] Zhandos A. Moldabekov, Michele Pavanello, Maximilian P. Böhme, Jan Vorberger, and Tobias Dornheim, "Linear-response time-dependent density functional theory approach to warm dense matter with adiabatic exchange-correlation kernels," Phys. Rev. Res. 5, 023089 (2023).

[40] Bastian Holst, Martin French, and Ronald Redmer, "Electronic transport coefficients from ab initio simulations and application to dense liquid hydrogen," Phys. Rev. B 83, 235120 (2011)

[41] Martin French, Gerd Röpke, Maximilian Schörner, Mandy Bethkenhagen, Michael P. Desjarlais, and Ronald Redmer, "Electronic transport coefficients from density functional theory across the plasma plane," Phys. Rev. E 105, 065204 (2022).

[42] V. V. Karasiev, L. Calderin, and S. B. Trickey, "Importance of finite-temperature exchange correlation for warm dense matter calculations," Phys. Rev. E 93, 063207 (2016).

[43] Valentin V. Karasiev, D. I. Mihaylov, and S. X. $\mathrm{Hu}$, "Meta-gga exchange-correlation free energy density functional to increase the accuracy of warm dense matter simulations," Phys. Rev. B 105, L081109 (2022).

[44] Raymond C. Clay, Jeremy Mcminis, Jeffrey M. McMahon, Carlo Pierleoni, David M. Ceperley, and Miguel A. Morales, "Benchmarking exchange-correlation functionals for hydrogen at high pressures using quantum monte carlo," Phys. Rev. B 89, 184106 (2014).

[45] Lars Goerigk, Andreas Hansen, Christoph Bauer, Stephan Ehrlich, Asim Najibi, and Stefan Grimme, "A look at the density functional theory zoo with the advanced gmtkn55 database for general main group thermochemistry, kinetics and noncovalent interactions," Phys. Chem. Chem. Phys. 19, 32184-32215 (2017).

[46] Valentin V. Karasiev, Travis Sjostrom, James Dufty, and S. B. Trickey, "Accurate homogeneous electron gas exchange-correlation free energy for local spin-density calculations," Phys. Rev. Lett. 112, 076403 (2014).

[47] S. Groth, T. Dornheim, T. Sjostrom, F. D. Malone, W. M. C. Foulkes, and M. Bonitz, "Ab initio exchangecorrelation free energy of the uniform electron gas at warm dense matter conditions," Phys. Rev. Lett. 119, 135001 (2017).

[48] Valentin V. Karasiev, James W. Dufty, and S. B. Trickey, "Nonempirical semilocal free-energy density functional for matter under extreme conditions," Phys. Rev. Lett. 120, 076401 (2018).

[49] John Kozlowski, Dennis Perchak, and Kieron Burke, "Generalized gradient approximation made thermal," (2023), arXiv:2308.03319 [physics.chem-ph].

[50] Zhandos Moldabekov, Maximilian Böhme, Jan Vorberger, David Blaschke, and Tobias Dornheim, "Ab initio static exchange-correlation kernel across jacob's ladder without functional derivatives," Journal of Chemical Theory and Computation 19, 1286-1299 (2023).

[51] S. H. Glenzer and R. Redmer, "X-ray thomson scattering in high energy density plasmas," Rev. Mod. Phys 81, 1625 (2009).

[52] D. H. Froula, J. S. Ross, L. Divol, and S. H. Glenzer, "Thomson-scattering techniques to diagnose local electron and ion temperatures, density, and plasma wave amplitudes in laser produced plasmas (invited)," Review of Scientific Instruments 77, 10E522 (2006).

[53] Tobias Dornheim, Maximilian Böhme, Dominik Kraus, Tilo Döppner, Thomas R. Preston, Zhandos A. Moldabekov, and Jan Vorberger, "Accurate temperature diagnostics for matter under extreme conditions," Nature Communications 13, 7911 (2022).

[54] G. Gregori, S. H. Glenzer, W. Rozmus, R. W. Lee, and O. L. Landen, "Theoretical model of x-ray scattering as a dense matter probe," Phys. Rev. E 67, 026412 (2003).

[55] K. Falk, S. P. Regan, J. Vorberger, B. J. B. Crowley, S. H. Glenzer, S. X. Hu, C. D. Murphy, P. B. Radha, A. P. Jephcoat, J. S. Wark, D. O. Gericke, and G. Gregori, "Comparison between x-ray scattering and velocity-interferometry measurements from shocked liquid deuterium," Phys. Rev. E 87, 043112 (2013).

[56] K. Falk, E. J. Gamboa, G. Kagan, D. S. Montgomery, B. Srinivasan, P. Tzeferacos, and J. F. Benage, "Equation of state measurements of warm dense carbon using laser-driven shock and release technique," Phys. Rev. Lett. 112, 155003 (2014).

[57] D Kraus, B Bachmann, B Barbrel, R W Falcone, L B Fletcher, S Frydrych, E J Gamboa, M Gauthier, D O Gericke, S H Glenzer, S Göde, E Granados, N J Hartley, J Helfrich, H J Lee, B Nagler, A Ravasio, W Schumaker, J Vorberger, and T Döppner, "Characterizing the ionization potential depression in dense carbon plasmas with high-precision spectrally resolved x-ray scattering," Plasma Physics and Controlled Fusion 61, 014015 (2018).

[58] J Chihara, "Difference in x-ray scattering between metallic and non-metallic liquids due to conduction electrons," Journal of Physics F: Metal Physics 17, 295-304 (1987).

[59] Zh. A. Moldabekov, T. Dornheim, M. Bonitz, and T. S. Ramazanov, "Ion energy-loss characteristics and friction in a free-electron gas at warm dense matter and nonideal dense plasma conditions," Phys. Rev. E 101, 053203 (2020).

[60] D. Casas, A.A. Andreev, M. Schnürer, M.D. BarrigaCarrasco, R. Morales, and L. González-Gallego, "Stopping power of a heterogeneous warm dense matter," Laser and Particle Beams 34, 306-314 (2016).

[61] M. Veysman, G. Röpke, M. Winkel, and H. Reinholz, "Optical conductivity of warm dense matter within a wide frequency range using quantum statistical and kinetic approaches," Phys. Rev. E 94, 013203 (2016).

[62] S. Jiang, O. L. Landen, H. D. Whitley, S. Hamel, R. London, D. S. Clark, P. Sterne, S. B. Hansen, S. X. Hu, G. W. Collins, and Y. Ping, "Thermal transport in warm dense matter revealed by refraction-enhanced xray radiography with a deep-neural-network analysis," Communications Physics 6, 98 (2023).

[63] N. M. Gill, C. J. Fontes, and C. E. Starrett, "Timedependent density functional theory applied to average atom opacity," Phys. Rev. E 103, 043206 (2021).

[64] P. Hollebon, O. Ciricosta, M. P. Desjarlais, C. Cacho, C. Spindloe, E. Springate, I. C. E. Turcu, J. S. Wark, and S. M. Vinko, "Ab initio simulations and measurements of the free-free opacity in aluminum," Phys. Rev. E 100, 043207 (2019).

[65] Xiaolei Zan, Chengliang Lin, Yong Hou, and Jianmin Yuan, "Local field correction to ionization potential depression of ions in warm or hot dense matter," Phys. Rev. E 104, 025203 (2021).

[66] G. Senatore, S. Moroni, and D. M. Ceperley, "Local field factor and effective potentials in liquid metals," J. Non-Cryst. Sol 205-207, 851-854 (1996).

[67] Zhandos Moldabekov, Tim Schoof, Patrick Ludwig, Michael Bonitz, and Tlekkabul Ramazanov, "Statically screened ion potential and Bohm potential in a quantum plasma," Physics of Plasmas 22, 102104 (2015).

[68] A. Pribram-Jones, P. E. Grabowski, and K. Burke, "Thermal density functional theory: Time-dependent linear response and approximate functionals from the fluctuation-dissipation theorem," Phys. Rev. Lett 116, 233001 (2016).

[69] Zhandos A. Moldabekov, Mani Lokamani, Jan Vorberger, Attila Cangi, and Tobias Dornheim, "Nonempirical mixing coefficient for hybrid XC functionals from analysis of the XC kernel," The Journal of Physical Chemistry Letters 14, 1326-1333 (2023), pMID: 36724891, https://doi.org/10.1021/acs.jpclett.2c03670.

[70] T. Dornheim, S. Groth, and M. Bonitz, "The uniform electron gas at warm dense matter conditions," Phys. Reports 744, 1-86 (2018).

[71] T. Dornheim, J. Vorberger, S. Groth, N. Hoffmann, Zh.A. Moldabekov, and M. Bonitz, "The static local field correction of the warm dense electron gas: An ab initio path integral Monte Carlo study and machine learning representation," J. Chem. Phys 151, 194104 (2019).

[72] Tobias Dornheim, Attila Cangi, Kushal Ramakrishna, Maximilian Böhme, Shigenori Tanaka, and Jan Vorberger, "Effective static approximation: A fast and reliable tool for warm-dense matter theory," Phys. Rev. Lett. 125, 235001 (2020).

[73] Tobias Dornheim, Zhandos A. Moldabekov, and Panagiotis Tolias, "Analytical representation of the local field correction of the uniform electron gas within the effective static approximation," Phys. Rev. B 103, 165102 (2021).

[74] Maximilian Böhme, Zhandos A. Moldabekov, Jan Vorberger, and Tobias Dornheim, "Static electronic density response of warm dense hydrogen: Ab initio path integral monte carlo simulations," Phys. Rev. Lett. 129, 066402 (2022).

[75] Maximilian Böhme, Zhandos A. Moldabekov, Jan Vorberger, and Tobias Dornheim, "Ab initio path integral monte carlo simulations of hydrogen snapshots at warm dense matter conditions," Phys. Rev. E 107, 015206 (2023).

[76] Minoru Takahashi and Masatoshi Imada, "Monte carlo calculation of quantum systems," Journal of the Physical Society of Japan 53, 963-974 (1984).

[77] D. M. Ceperley, "Path integrals in the theory of condensed helium," Rev. Mod. Phys 67, 279 (1995).

[78] M. Boninsegni, N. V. Prokofev, and B. V. Svistunov, "Worm algorithm and diagrammatic Monte Carlo: A new approach to continuous-space path integral Monte Carlo simulations," Phys. Rev. E 74, 036701 (2006).

[79] Zhandos Moldabekov, Tobias Dornheim, Jan Vorberger, and Attila Cangi, "Benchmarking exchange-correlation functionals in the spin-polarized inhomogeneous electron gas under warm dense conditions," Phys. Rev. B 105,035134 (2022).

[80] Zhandos Moldabekov, Sebastian Schwalbe, Maximilian P. Böhme, Jan Vorberger, Xuecheng Shao, Michele Pavanello, Frank R. Graziani, and Tobias Dornheim, "Bound-state breaking and the importance of thermal exchange-correlation effects in warm dense hydrogen," Journal of Chemical Theory and Computation 20, 6878 (2024).

[81] D. M. Ceperley, "Fermion nodes," Journal of Statistical Physics 63, 1237-1267 (1991).

[82] T. Dornheim, "Fermion sign problem in path integral Monte Carlo simulations: Quantum dots, ultracold atoms, and warm dense matter," Phys. Rev. E 100, 023307 (2019).

[83] M. Troyer and U. J. Wiese, "Computational complexity and fundamental limitations to fermionic quantum Monte Carlo simulations," Phys. Rev. Lett 94, 170201 (2005).

[84] Tobias Dornheim, Panagiotis Tolias, Simon Groth, Zhandos A. Moldabekov, Jan Vorberger, and Barak Hirshberg, "Fermionic physics from ab initio path integral Monte Carlo simulations of fictitious identical particles," The Journal of Chemical Physics 159, 164113 (2023).

[85] Tobias Dornheim, Sebastian Schwalbe, Zhandos A. Moldabekov, Jan Vorberger, and Panagiotis Tolias, "Ab initio path integral Monte Carlo simulations of the uniform electron gas on large length scales," J. Phys. Chem. Lett. 15, 1305-1313 (2024).

[86] Yunuo Xiong and Hongwei Xiong, "On the thermodynamic properties of fictitious identical particles and the application to fermion sign problem," The Journal of Chemical Physics 157, 094112 (2022).

[87] Tobias Dornheim, Tilo Döppner, Panagiotis Tolias, Maximilian Böhme, Luke Fletcher, Thomas Gawne, Frank Graziani, Dominik Kraus, Michael MacDonald, Zhandos Moldabekov, Sebastian Schwalbe, Dirk Gericke, and Jan Vorberger, "Unraveling electronic correlations in warm dense quantum plasmas," (2024), arXiv:2402.19113 [physics.plasm-ph].

[88] Tobias Dornheim, Sebastian Schwalbe, Maximilian Böhme, Zhandos Moldabekov, Jan Vorberger, and Panagiotis Tolias, "Ab initio path integral monte carlo simulations of warm dense two-component systems without fixed nodes: structural properties," (2024), arXiv:2403.01979 [physics.comp-ph].

[89] T. Dornheim, S. Groth, T. Sjostrom, F. D. Malone, W. M. C. Foulkes, and M. Bonitz, "Ab initio quantum Monte Carlo simulation of the warm dense electron gas in the thermodynamic limit," Phys. Rev. Lett. 117, 156403 (2016).

[90] Tobias Dornheim and Jan Vorberger, "Overcoming finite-size effects in electronic structure simulations at extreme conditions," The Journal of Chemical Physics 154, 144103 (2021).

[91] U. Zastrau, P. Sperling, M. Harmand, A. Becker, T. Bornath, R. Bredow, S. Dziarzhytski, T. Fennel, L. B. Fletcher, E. F" orster, S. G"ode, G. Gregori, V. Hilbert, D. Hochhaus, B. Holst, T. Laarmann, H. J. Lee, T. Ma, J. P. Mithen, R. Mitzner, C. D. Murphy, M. Nakatsutsumi, P. Neumayer, A. Przystawik, S. Roling, M. Schulz, B. Siemer, S. Skruszewicz, J. Tiggesb"aumker, S. Toleikis, T. Tschentscher, T. White, M. W" ostmann, H. Zacharias, T. D" oppner, S. H. Glenzer, and R. Redmer, "Resolving ultrafast heating of dense cryogenic hydrogen," Phys. Rev. Lett 112, 105002 (2014).

[92] L. B. Fletcher, J. Vorberger, W. Schumaker, C. Ruyer, S. Goede, E. Galtier, U. Zastrau, E. P. Alves, S. D. Baalrud, R. A. Baggott, B. Barbrel, Z. Chen, T. Döppner, M. Gauthier, E. Granados, J. B. Kim, D. Kraus, H. J. Lee, M. J. MacDonald, R. Mishra, A. Pelka, A. Ravasio, C. Roedel, A. R. Fry, R. Redmer, F. Fiuza, D. O. Gericke, and S. H. Glenzer, "Electron-ion temperature relaxation in warm dense hydrogen observed with picosecond resolved x-ray scattering," Frontiers in Physics 10 (2022), 10.3389/fphy.2022.838524.

[93] Maximilian P. Böhme, Luke B. Fletcher, Tilo Döppner, Dominik Kraus, Andrew D. Baczewski, Thomas R. Preston, Michael J. MacDonald, Frank R. Graziani, Zhandos A. Moldabekov, Jan Vorberger, and Tobias Dornheim, "Evidence of free-bound transitions in warm dense matter and their impact on equation-of-state measurements," (2023), arXiv:2306.17653 [physics.plasm$\mathrm{ph}]$.

[94] Louisa M. Fraser, W. M. C. Foulkes, G. Rajagopal, R. J. Needs, S. D. Kenny, and A. J. Williamson, "Finitesize effects and coulomb interactions in quantum monte carlo calculations for homogeneous systems with periodic boundary conditions," Phys. Rev. B 53, 1814-1832 (1996).

[95] M. F. Herman, E. J. Bruskin, and B. J. Berne, "On path integral monte carlo simulations," The Journal of Chemical Physics 76, 5150-5155 (1982).

[96] David Chandler and Peter G. Wolynes, "Exploiting the isomorphism between quantum theory and classical statistical mechanics of polyatomic fluids," The Journal of Chemical Physics 74, 4078-4095 (1981).

[97] Nicholas Metropolis, Arianna W. Rosenbluth, Marshall N. Rosenbluth, Augusta H. Teller, and Edward Teller, "Equation of state calculations by fast computing machines," The Journal of Chemical Physics 21, 1087-1092 (1953).

[98] H. Kleinert, Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets, EBL-Schweitzer (World Scientific, 2009).

[99] B. Militzer, "Computation of the high temperature coulomb density matrix in periodic boundary conditions," Computer Physics Communications 204, 88-96 (2016).

[100] Ethan W. Brown, Bryan K. Clark, Jonathan L. DuBois, and David M. Ceperley, "Path-integral monte carlo simulation of the warm dense homogeneous electron gas," Phys. Rev. Lett. 110, 146405 (2013).

[101] Tobias Dornheim, Simon Groth, Alexey Filinov, and Michael Bonitz, "Permutation blocking path integral monte carlo: a highly efficient approach to the simulation of strongly degenerate non-ideal fermions," New Journal of Physics 17, 073017 (2015).

[102] Burkhard Militzer and Kevin P. Driver, "Development of path integral monte carlo simulations with localized nodal surfaces for second-row elements," Phys. Rev. Lett. 115, 176403 (2015).

[103] Tobias Dornheim, Michele Invernizzi, Jan Vorberger, and Barak Hirshberg, "Attenuating the fermion sign problem in path integral monte carlo simulations using the bogoliubov inequality and thermodynamic integration," The Journal of Chemical Physics 153, 234104 (2020).

[104] Barak Hirshberg, Michele Invernizzi, and Michele Parrinello, "Path integral molecular dynamics for fermions: Alleviating the sign problem with the bogoliubov inequality," The Journal of Chemical Physics 152, 171102 (2020).

[105] A. Filinov and M. Bonitz, "Collective and single-particle excitations in two-dimensional dipolar bose gases," Phys. Rev. A 86, 043628 (2012).

[106] Eran Rabani, David R. Reichman, Goran Krilov, and Bruce J. Berne, "The calculation of transport properties in quantum liquids using the maximum entropy numerical analytic continuation method: Application to liquid para-hydrogen," Proceedings of the National Academy of Sciences 99, 1129-1133 (2002).

[107] Tobias Dornheim, Zhandos A. Moldabekov, and Jan Vorberger, "Nonlinear density response from imaginarytime correlation functions: Ab initio path integral monte
carlo simulations of the warm dense electron gas," The Journal of Chemical Physics 155, 054110 (2021).

[108] Tobias Dornheim, Zhandos Moldabekov, Panagiotis Tolias, Maximilian Böhme, and Jan Vorberger, "Physical insights from imaginary-time density-density correlation functions," Matter and Radiation at Extremes 8, 056601 (2023).

[109] Tobias Dornheim, Jan Vorberger, Zhandos A. Moldabekov, and Maximilian Böhme, "Analysing the dynamic structure of warm dense matter in the imaginarytime domain: theoretical models and simulations," Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 381, 20220217 (2023)

[110] Tobias Dornheim, Maximilian P. Böhme, David A. Chapman, Dominik Kraus, Thomas R. Preston, Zhandos A. Moldabekov, Niclas Schlünzen, Attila Cangi, Tilo Döppner, and Jan Vorberger, "Imaginary-time correlation function thermometry: A new, high-accuracy and model-free temperature analysis technique for $\mathrm{x}$ ray Thomson scattering data," Physics of Plasmas 30, 042707 (2023).

[111] Tobias Dornheim, Tilo Döppner, Andrew D. Baczewski, Panagiotis Tolias, Maximilian P. Böhme, Zhandos A. Moldabekov, Divyanshu Ranjan, David A. Chapman, Michael J. MacDonald, Thomas R. Preston, Dominik Kraus, and Jan Vorberger, "X-ray thomson scattering absolute intensity from the f-sum rule in the imaginary-time domain," arXiv (2023), 2305.15305 [physics.plasm-ph].

[112] Setsuo Ichimaru, Hiroshi Iyetomi, and Shigenori Tanaka, "Statistical physics of dense plasmas: Thermodynamics, transport coefficients and dynamic correlations," Physics Reports 149, 91-205 (1987).

[113] Paul Hamann, Tobias Dornheim, Jan Vorberger, Zhandos A. Moldabekov, and Michael Bonitz, "Dynamic properties of the warm dense electron gas based on abinitio path integral monte carlo simulations," Phys. Rev. B 102, 125150 (2020).

[114] Paul Hamann, Jan Vorberger, Tobias Dornheim, Zhandos A. Moldabekov, and Michael Bonitz, "Ab initio results for the plasmon dispersion and damping of the warm dense electron gas," Contributions to Plasma Physics 60, e202000147 (2020).

[115] C. Ullrich, Time-Dependent Density-Functional Theory: Concepts and Applications, Oxford Graduate Texts (OUP Oxford, 2012).

[116] S. Moroni, D. M. Ceperley, and G. Senatore, "Static response from quantum Monte Carlo calculations," Phys. Rev. Lett 69, 1837 (1992).

[117] S. Moroni, D. M. Ceperley, and G. Senatore, "Static response and local field factor of the electron gas," Phys. Rev. Lett 75, 689 (1995).

[118] T. Dornheim, S. Groth, J. Vorberger, and M. Bonitz, "Permutation blocking path integral Monte Carlo approach to the static density response of the warm dense electron gas," Phys. Rev. E 96, 023203 (2017).

[119] S. Groth, T. Dornheim, and M. Bonitz, "Configuration path integral Monte Carlo approach to the static density response of the warm dense electron gas," J. Chem. Phys 147, 164108 (2017)

[120] Panagiotis Tolias, Tobias Dornheim, Zhandos Moldabekov, and Jan Vorberger, "Unravelling the nonlinear ideal density response of many-body systems," EPL
142, 44001 (2023).

[121] Tobias Dornheim, Jan Vorberger, and Michael Bonitz, "Nonlinear electronic density response in warm dense matter," Phys. Rev. Lett. 125, 085001 (2020).

[122] Zhandos Moldabekov, Tobias Dornheim, Maximilian Böhme, Jan Vorberger, and Attila Cangi, "The relevance of electronic perturbations in the warm dense electron gas," The Journal of Chemical Physics 155, 124116 (2021).

[123] Zhandos A. Moldabekov, Xuecheng Shao, Michele Pavanello, Jan Vorberger, Frank Graziani, and Tobias Dornheim, "Imposing correct jellium response is key to predict the density response by orbital-free DFT," Phys. Rev. B 108, 235168 (2023).

[124] S. Groth, T. Dornheim, and J. Vorberger, "Ab initio path integral Monte Carlo approach to the static and dynamic density response of the uniform electron gas," Phys. Rev. B 99, 235122 (2019).

[125] Tobias Dornheim, Travis Sjostrom, Shigenori Tanaka, and Jan Vorberger, "Strongly coupled electron liquid: $\mathrm{Ab}$ initio path integral monte carlo simulations and dielectric theories," Phys. Rev. B 101, 045129 (2020).

[126] Tobias Dornheim, Zhandos A Moldabekov, Jan Vorberger, and Simon Groth, "Ab initio path integral monte carlo simulation of the uniform electron gas in the high energy density regime," Plasma Physics and Controlled Fusion 62, 075003 (2020).

[127] P. Tolias, F. Lucco Castello, and T. Dornheim, "Integral equation theory based dielectric scheme for strongly coupled electron liquids," The Journal of Chemical Physics 155, 134115 (2021).

[128] Tobias Dornheim, Jan Vorberger, Zhandos Moldabekov, Gerd Röpke, and Wolf-Dietrich Kraeft, "The uniform electron gas at high temperatures: ab initio path integral monte carlo simulations and analytical theory," High Energy Density Physics 45, 101015 (2022).

[129] Tobias Dornheim, Maximilian Böhme, Zhandos A. Moldabekov, Jan Vorberger, and Michael Bonitz, "Density response of the warm dense electron gas beyond linear response theory: Excitation of harmonics," Phys. Rev. Research 3, 033231 (2021).

[130] Tobias Dornheim, Panagiotis Tolias, Zhandos A. Moldabekov, and Jan Vorberger, "Energy response and spatial alignment of the perturbed electron gas," The Journal of Chemical Physics 158, 164108 (2023).

[131] A. A. Kugler, "Theory of the local field correction in an electron gas," J. Stat. Phys 12, 35 (1975).

[132] Kazutami Tago, Kenichi Utsumi, and Setsuo Ichimaru, "Local Field Effect in Strongly Coupled, Classical OneComponent Plasma," Progress of Theoretical Physics 65, 54-65 (1981).

[133] Tobias Dornheim, Jan Vorberger, Zhandos A. Moldabekov, and Panagiotis Tolias, "Spin-resolved density response of the warm dense electron gas," Phys. Rev. Research 4, 033018 (2022).

[134] D. Kremp, M. Schlanges, and W.-D. Kraeft, Quantum Statistics of Nonideal Plasmas (Springer, Heidelberg, 2005).

[135] M. J. MacDonald, A. M. Saunders, B. Bachmann, M. Bethkenhagen, L. Divol, M. D. Doyle, L. B. Fletcher, S. H. Glenzer, D. Kraus, O. L. Landen, H. J. LeFevre, S. R. Klein, P. Neumayer, R. Redmer, M. Schörner, N. Whiting, R. W. Falcone, and T. Döppner, "Demonstration of a laser-driven, narrow spectral bandwidth x-
ray source for collective x-ray scattering experiments," Physics of Plasmas 28, 032708 (2021).

[136] E. Vitali, M. Rossi, L. Reatto, and D. E. Galli, "Ab initio low-energy dynamics of superfluid and solid ${ }^{4} \mathrm{He}, "$ Phys. Rev. B 82, 174510 (2010).

[137] J. Vorberger and D. O. Gericke, "Ab initio approach to model x-ray diffraction in warm dense matter," Phys. Rev. E 91, 033112 (2015).

[138] T. Döppner, M. Bethkenhagen, D. Kraus, P. Neumayer, D. A. Chapman, B. Bachmann, R. A. Baggott, M. P. Böhme, L. Divol, R. W. Falcone, L. B. Fletcher, O. L. Landen, M. J. MacDonald, A. M. Saunders, M. Schörner, P. A. Sterne, J. Vorberger, B. B. L. Witte, A. Yi, R. Redmer, S. H. Glenzer, and D. O. Gericke, "Observing the onset of pressure-driven k-shell delocalization," Nature 618, 270-275 (2023).

[139] Tobias Dornheim, Maximilian Böhme, Burkhard Militzer, and Jan Vorberger, "Ab initio path integral monte carlo approach to the momentum distribution of the uniform electron gas at finite temperature without fixed nodes," Phys. Rev. B 103, 205142 (2021).

[140] Tobias Dornheim, Maximilian Böhme, and Sebastian Schwalbe, "ISHTAR - Imaginary-time Stochastic Highperformance Tool for Ab initio Research," (2024).

[141] M. Boninsegni, N. V. Prokofev, and B. V. Svistunov, "Worm algorithm for continuous-space path integral Monte Carlo simulations," Phys. Rev. Lett 96, 070601 (2006).

[142] A link to a repository containing all PIMC raw data will be made available upon publication.

[143] Tobias Dornheim, Damar C. Wicaksono, Juan E. Suarez-Cardona, Panagiotis Tolias, Maximilian P. Böhme, Zhandos A. Moldabekov, Michael Hecht, and Jan Vorberger, "Extraction of the frequency moments of spectral densities from imaginary-time correlation function data," Phys. Rev. B 107, 155148 (2023).

[144] T. Dornheim, S. Groth, J. Vorberger, and M. Bonitz, "Ab initio path integral Monte Carlo results for the dynamic structure factor of correlated electrons: From the electron liquid to warm dense matter," Phys. Rev. Lett. 121,255001 (2018).
[145] A. A. Kugler, "Bounds for some equilibrium properties of an electron gas," Phys. Rev. A 1, 1688 (1970).

[146] Tobias Dornheim, Maximilian P. Böhme, Zhandos A. Moldabekov, and Jan Vorberger, "Electronic density response of warm dense hydrogen on the nanoscale," Phys. Rev. E 108, 035204 (2023).

[147] S. Mazevet, M. P. Desjarlais, L. A. Collins, J. D. Kress, and N. H. Magee, "Simulations of the optical properties of warm dense aluminum," Phys. Rev. E 71, 016409 (2005).

[148] M. P. Desjarlais, J. D. Kress, and L. A. Collins, "Electrical conductivity for warm, dense aluminum plasmas and liquids," Phys. Rev. E 66, 025401(R) (2002).

[149] Tobias Dornheim, Zhandos Moldabekov, Jan Vorberger, Hanno Kählert, and Michael Bonitz, "Electronic pair alignment and roton feature in the warm dense electron gas," Communications Physics 5, 304 (2022).

[150] Paul Hamann, Linda Kordts, Alexey Filinov, Michael Bonitz, Tobias Dornheim, and Jan Vorberger, "Prediction of a roton-type feature in warm dense hydrogen," Phys. Rev. Res. 5, 033039 (2023).

[151] Carl A. Kukkonen and Kun Chen, "Quantitative electron-electron interaction using local field factors from quantum monte carlo calculations," Phys. Rev. B 104, 195142 (2021).

[152] Tobias Dornheim, Panagiotis Tolias, Zhandos A. Moldabekov, Attila Cangi, and Jan Vorberger, "Effective electronic forces and potentials from ab initio path integral Monte Carlo simulations," The Journal of Chemical Physics 156, 244113 (2022).

[153] A. D. Baczewski, L. Shulenburger, M. P. Desjarlais, S. B. Hansen, and R. J. Magyar, "X-ray thomson scattering in warm dense matter without the chihara decomposition," Phys. Rev. Lett. 116, 115004 (2016).

[154] Mark Jarrell and J.E. Gubernatis, "Bayesian inference and the analytic continuation of imaginary-time quantum monte carlo data," Physics Reports 269, 133-195 (1996).

[155] Yunuo Xiong and Hongwei Xiong, "Thermodynamics of fermions at any temperature based on parametrized partition function," Phys. Rev. E 107, 055308 (2023).


[^0]:    * t.dornheim@hzdr.de

</end of paper 3>


<paper 4>
# Dynamic exchange-correlation effects in the strongly coupled electron liquid 

Tobias Dornheim, ${ }^{1,2, *}$ Panagiotis Tolias, ${ }^{3}$ Fotios Kalkavouras, ${ }^{3}$ Zhandos A. Moldabekov, ${ }^{1,2}$ and Jan Vorberger ${ }^{2}$<br>${ }^{1}$ Center for Advanced Systems Understanding (CASUS), D-02826 Görlitz, Germany<br>${ }^{2}$ Helmholtz-Zentrum Dresden-Rossendorf (HZDR), D-01328 Dresden, Germany<br>${ }^{3}$ Space and Plasma Physics, Royal Institute of Technology (KTH), Stockholm, SE-100 44, Sweden

We present the first quasi-exact $a b$ initio path integral Monte Carlo (PIMC) results for the dynamic local field correction $\widetilde{G}\left(\mathbf{q}, z_{l} ; r_{s}, \Theta\right)$ in the imaginary Matsubara frequency domain, focusing on the strongly coupled finite temperature uniform electron gas. These allow us to investigate the impact of dynamic exchange-correlation effects onto the static structure factor. Our results provide a straightforward explanation for previously reported spurious effects in the so-called static approximation [Dornheim et al., Phys. Rev. Lett. 125, 235001 (2020)], where the frequency-dependence of the local field correction is neglected. Our findings hint at the intriguing possibility of constructing an analytical four-parameter representation of $\widetilde{G}\left(\mathbf{q}, z_{l} ; r_{s}, \Theta\right)$ valid for a substantial part of the phase diagram, which would constitute key input for thermal density functional theory simulations.

## I. INTRODUCTION

The uniform electron gas (UEG) [1-3] constitutes the archetypal model system of interacting electrons in physics, quantum chemistry, and related fields. Having originally been introduced as a qualitative description of the conduction electrons in simple metals [4], the UEG exhibits a surprising wealth of intrinsically interesting physical effects such as the roton-type negative dispersion relation of its dynamic structure factor [5-10] and Wigner crystallization at very low densities [11-13]. In addition, fast and reliable parametrizations [14-18] of highly accurate quantum Monte Carlo simulations in the electronic ground state [19-22] have been of paramount importance for the remarkable success of density functional theory (DFT) [23], and a host of other applications.

Very recently, the high interest in inertial confinement nuclear fusion experiments [24-26] and dense astrophysical bodies (giant planet interiors [27], brown dwarfs [28], white dwarf atmospheres [29]) has triggered similar developments [30-36] in the warm dense matter (WDM) regime $[3,37,38]$, which is characterized by the complex interplay of Coulomb coupling, thermal excitations, and partially degenerate electrons. The rigorous description of these extreme states thus requires a holistic description, which is challenging [37]. In practice, the most accurate method is given by the $a b$ initio path integral Monte Carlo (PIMC) approach [39, 40], which is capable of delivering quasi-exact results for $\Theta \gtrsim 1$, with $\Theta=k_{\mathrm{B}} T / E_{\mathrm{F}}$ the reduced temperature [41] and $E_{\mathrm{F}}$ the Fermi energy.

A fundamental field of investigation concerns the linear density response of the UEG [42], which is fully characterized by the dynamic density response function [43]

$$
\begin{equation*}
\chi(\mathbf{q}, \omega)=\frac{\chi_{0}(\mathbf{q}, \omega)}{1-4 \pi / q^{2}[1-G(\mathbf{q}, \omega)] \chi_{0}(\mathbf{q}, \omega)} \tag{1}
\end{equation*}
$$

where $\chi_{0}(\mathbf{q}, \omega)$ is the Lindhard function describing the density response of an ideal (i.e., noninteracting) Fermi[^0]

gas at the same density and temperature. Note that Hartree atomic units are employed throughout this work. Setting $G(\mathbf{q}, \omega) \equiv 0$ in Eq. (1) corresponds to the random phase approximation (RPA), where the density response is described on the mean-field level. Therefore, the full wave-vector and frequency resolved information about electronic exchange-correlation (XC) effects is contained in the dynamic local field correction (LFC) $G(\mathbf{q}, \omega)$. Naturally, the LFC constitutes important input for a gamut of applications, including the estimation of ionization potential depression [44], calculation of the stopping power [45], and interpretation of X-ray Thomson scattering (XRTS) measurements [46]. Moreover, it is directly related to the dynamic XC-kernel $K_{\mathrm{xc}}(\mathbf{q}, \omega)$, which is the key quantity in linear-response time-dependent density functional theory (LR-TDDFT) simulations [47-49].

A particularly useful result of linear response theory is the well-known fluctuation-dissipation theorem [1]

$$
\begin{equation*}
S(\mathbf{q}, \omega)=-\frac{\operatorname{Im} \chi(\mathbf{q}, \omega)}{\pi n\left(1-e^{-\beta \omega}\right)} \tag{2}
\end{equation*}
$$

where $n$ is the electronic number density and $\beta=1 / k_{\mathrm{B}} T$ is the inverse temperature in energy units. It relates $\chi(\mathbf{q}, \omega)$ - an effective single-electron property - with the dynamic structure factor $S(\mathbf{q}, \omega)$, that is defined as the Fourier transform of the intermediate scattering function $F(\mathbf{q}, t)=\langle\hat{n}(\mathbf{q}, t) \hat{n}(-\mathbf{q}, 0)\rangle[50]$ and, thus, constitutes an electron-electron correlation function. We note that $S(\mathbf{q}, \omega)$ is routinely (though indirectly due to the inevitable convolution with the source-and-instrument function) probed in scattering experiments [50]; for example, XRTS measurements are a key diagnostic technique for experiments with high energy density matter [51-57]. In addition, an integration over the frequency yields the static structure factor (SSF)

$$
\begin{equation*}
S(\mathbf{q})=\int_{-\infty}^{\infty} \mathrm{d} \omega S(\mathbf{q}, \omega) \tag{3}
\end{equation*}
$$

which is connected to the electronic pair correlation function $g(\mathbf{r})$. Finally, the combination of Eqs. (1), (2) and (3) with the adiabatic connection formula gives one access to the XC-free energy of a given system, which can
be used as a promising route for the construction of advanced, nonlocal and explicitly thermal XC-functionals for DFT simulations [58].

In practice, obtaining accurate results for the dynamic LFC of the UEG is difficult. Consequently, previous efforts were mostly based on various approximations and interpolations, e.g. [59-64]. Fairly recently, Dornheim et al. $[6,7,65,66]$ have presented the first highly accurate results for $S(\mathbf{q}, \omega)$ based on the analytic continuation $[67]$ of the imaginary-time density-density correlation function (ITCF) $F(\mathbf{q}, \tau)=\langle\hat{n}(\mathbf{q}, \tau) \hat{n}(-\mathbf{q}, 0)\rangle$, cf. Eq. (5) below. This was achieved through the stochastic sampling of $G(\mathbf{q}, \omega)$, which has rendered the analytic continuation practical in the case of the UEG. For completeness, we note that very recently LeBlanc et al. [68] have presented complementary results for lower temperatures. Remarkably, it has been reported [6] that the static approximation, i.e., setting $G(\mathbf{q}, \omega) \equiv G(\mathbf{q}, 0)$ in Eq. (1) gives very accurate results for $S(\mathbf{q}, \omega)$ and related observables $[66,69]$ over a broad range of parameters. Unfortunately, the situation turned out to be somewhat more complex: while observables such as $S(\mathbf{q}, \omega)$ are indeed very accurate for a given wave vector, integrating over $\mathbf{q}$ as it is required to estimate e.g. the interaction energy leads to a substantial accumulation of individual small errors [70]; these originate from a systematic overestimation of $S(\mathbf{q})$ for large wave numbers that is connected to the diverging on-top pair correlation function $g(0)$.

In lieu of a full dynamic LFC, it was subsequently suggested to replace the exact static limit of $G(\mathbf{q}, \omega)$ by an effectively frequency-averaged and, therefore, inherently static LFC $G(\mathbf{q})$. The resulting effective static approximation (ESA) combines available PIMC results for $G(\mathbf{q}, 0)[71]$ with a parametrization of restricted PIMC results for $g(0)$ [72] and removes the deficiencies of the static approximation by design. While being computationally cheap and easy to use for various applications, the ESA scheme is somewhat empirical. Furthermore, it is incapable of providing detailed insights concerning the actual importance of dynamic XC-effects, since the entire frequency dependence has been averaged out for $g(0)$.

To overcome this limitation, we first represent $S(\mathbf{q})$ as a Matsubara series [73]

$$
\begin{equation*}
S(\mathbf{q})=-\frac{1}{n \beta} \sum_{l=-\infty}^{\infty} \widetilde{\chi}\left(\mathbf{q}, z_{l}\right) \tag{4}
\end{equation*}
$$

where $z_{l}=i 2 \pi l / \beta$ are the imaginary bosonic Matsubara frequencies and where the tilde symbol signifies dynamic quantities whose definition has been extended in the complex frequency domain by means of analytic continuation. We note that Eq.(4) is a key expression in the finite temperature dielectric formalism [73-83]. To unambiguously resolve the impact of dynamic XC-effects, we utilize the very recent Fourier-Matsubara series representation of the ITCF $F(\mathbf{q}, \tau)$ derived by Tolias et al. [84] to obtain quasi-exact PIMC results for the dynamic LFC $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ of the UEG in the discrete Matsubara frequency domain. Our results provide new insights into the validity of the
ESA and the deficiencies of the static approximation, and further elucidate the complex interplay of quantum delocalization with electronic XC-effects. Moreover, the presented approach opens up new opportunities for the systematic development of fully dynamic dielectric theories and the construction of improved XC-functionals for DFT simulations of WDM and beyond.

The paper is organized as follows. In Sec. II, we introduce the theoretical background, including the imaginary-time correlation functions (II A), the exact Fourier-Matsubara expansion (IIB), the exact high frequency behavior of the dynamic LFC (II C), and the effective static approximation (II D). In Sec. III, we present our new results for the dynamic density response (III A), for the impact of dynamic XC-effects onto the static structure factor (IIIB), and an analysis of the highfrequency limit of the LFC (IIIC). The paper is concluded by a summary and outlook in Sec. IV.

## II. THEORY

The $a b$ initio PIMC method is based on the Feynman imaginary-time path integral formalism of statistical mechanics [85], which exploits the isomorphism between the canonical density $\hat{\varrho}=e^{-\beta \hat{H}}$ and time evolution operators within an interval $t=-i \hbar \beta$. Detailed derivation of the method [39], information on efficient path sampling schemes [86-88], and discussion of the estimation of various imaginary-time correlation functions [89] has been presented in the literature and need not be repeated here.

## A. Imaginary-time correlation function

The PIMC method is capable of providing asymptotically exact estimates for any well-defined thermodynamic observable. This includes integrated quantities such as the pressure and the energies, various many-body correlation functions, and even the off-diagonal density matrix and the momentum distribution [88, 90, 91]. Moreover, PIMC gives one access to the full equilibrium dynamics of the simulated system, but in the imaginary-time domain. There exist a variety of interesting imaginary-time correlation functions that are connected with dynamic properties; for example, the Matsubara Green function is related to the well-known single-particle spectral function $A(\mathbf{q}, \omega)$. In the present work, we focus on the ITCF $F(\mathbf{q}, \tau)$ introduced above.

The ITCF $F(\mathbf{q}, \tau)$ is connected to the dynamic structure factor $S(\mathbf{q}, \omega)$ via a two-sided Laplace transform [6]

$$
\begin{equation*}
F(\mathbf{q}, \tau)=\mathcal{L}[S(\mathbf{q}, \omega)]=\int_{-\infty}^{\infty} \mathrm{d} \omega S(\mathbf{q}, \omega) e^{-\tau \omega} \tag{5}
\end{equation*}
$$

In principle, the Laplace transform constitutes a unique mapping and $F(\mathbf{q}, \tau)$ contains, by definition, the same information as $S(\mathbf{q}, \omega)$, only in a different representation. Nevertheless, the numerical inversion of Eq. (5) to
solve for $S(\mathbf{q}, \omega)$ is an ill-posed problem [67], and the associated difficulties are further exacerbated by the error bars in the ITCF. The inversion problem was successfully overcome for the UEG, for parts of the WDM regime, based on the stochastic sampling of $G(\mathbf{q}, \omega)$ taking into account a number of rigorous constraints $[6,7,66]$. Similar relations that connect higher-order dynamic structure factors with higher-order imaginary-time correlation functions have been reported in the literature [89].

Another important application of the ITCF is given by the imaginary-time version of the fluctuation-dissipation theorem [92], which relates $F(\mathbf{q}, \tau)$ to the static limit of the density response function, see Eq. (1),

$$
\begin{equation*}
\chi(\mathbf{q}, 0)=-2 n \int_{0}^{\beta / 2} \mathrm{~d} \tau F(\mathbf{q}, \tau) \tag{6}
\end{equation*}
$$

In practice, Eq. (6) implies that it is possible to get the full wave number dependence of the static linear density response from a single simulation of the unperturbed system. This relation provided the basis for a number of investigations of the uniform electron gas covering a broad range of parameters $[7,71,77,79,93,94]$. These efforts have recently been extended to the case of warm dense hydrogen [95], providing the first results for the speciesresolved static local field factors of a real WDM system. Moreover, there exist nonlinear generalizations of Eq. (6) that relate higher-order ITCFs with different nonlinear density response functions, see Refs. [42, 89, 96].

Furthermore, we point out that the ITCF has emerged as an important quantity in the interpretation of XRTS experiments $[52,53,97-101]$, as the deconvolution with respect to the source-and-instrument function is substantially more stable in the Laplace domain compared to the usual frequency domain. For example, the symmetry relation $F(\mathbf{q}, \tau)=F(\mathbf{q}, \beta-\tau)$ gives one model-free access to the temperature for arbitrarily complicated materials in thermal equilibrium.

For completeness, we note that the fixed node approximation [102] that is often imposed to circumvent the numerical fermion sign problem [103] breaks the usual imaginary-time translation invariance of PIMC, and thus access to imaginary-time properties is lost. Hence, we use the direct PIMC method throughout this work, making our simulations computationally costly, but exact within the statistical error bars.

## B. Fourier-Matsubara expansion

Tolias and coworkers recently derived an exact FourierMatsubara series expansion for the ITCF that reads [84]

$$
\begin{equation*}
F(\boldsymbol{q}, \tau)=-\frac{1}{n \beta} \sum_{l=-\infty}^{+\infty} \widetilde{\chi}\left(q, z_{l}\right) e^{-z_{l} \tau} \tag{7}
\end{equation*}
$$

which constitutes the generalization of the Matsubara series for the SSF, see Eq.(4), since $F(\boldsymbol{q}, \tau=0)=S(\boldsymbol{q})$.
The coefficients of the Fourier-Matsubara series are given by $[84]$

$$
\begin{equation*}
\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)=-2 n \int_{0}^{\beta / 2} \mathrm{~d} \tau F(\mathbf{q}, \tau) \cos \left(i z_{l} \tau\right) \tag{8}
\end{equation*}
$$

which constitutes the generalization of the imaginarytime version of the fluctuation-dissipation theorem, see Eq.(6), since $\widetilde{\chi}\left(\boldsymbol{q}, z_{l} \rightarrow 0\right)=\chi(\mathbf{q}, 0)$. Finally, solving Eq. (1) for the dynamic Matsubara LFC, we obtain

$$
\begin{equation*}
\widetilde{G}\left(\mathbf{q}, z_{l}\right)=1-\frac{q^{2}}{4 \pi}\left[\frac{1}{\widetilde{\chi}_{0}\left(\mathbf{q}, z_{l}\right)}-\frac{1}{\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)}\right] \tag{9}
\end{equation*}
$$

Utilizing these expressions, in order to characterize the impact of dynamic XC-effects, we switch to the Matsubara imaginary frequency domain. We extract the dynamic Matsubara density response function $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ from our highly accurate PIMC results for the ITCF via Eq. (8), we compute the SSF via the Matsubara series of Eq. (4), and we calculate the dynamic Matsubara LFC via Eq. (9). Such a dynamic Matsubara LFC extraction becomes problematic at high Matsubara orders and at high wave numbers. Thus, exact asymptotic results need to be invoked.

## C. Exact high frequency behavior of the Matsubara local field correction

The spectral representation of the dynamic LFC formally extends its domain of definition from real frequencies $\omega$ to complex frequencies $z$. It reads [43]

$$
\begin{equation*}
\widetilde{G}(\mathbf{q}, z)=G(\mathbf{q}, \infty)-\frac{1}{\pi} \int_{-\infty}^{+\infty} \frac{\Im\{G(\mathbf{q}, \omega)\}}{z-\omega} d \omega \tag{10}
\end{equation*}
$$

where $G(\mathbf{q}, \infty)=\Re\{G(\mathbf{q}, \infty)\}$ due to $\Im\{G(\mathbf{q}, \infty)\}=0[1]$. This directly leads to $\widetilde{G}(\mathbf{q}, \imath \infty)=G(\mathbf{q}, \infty)$ for $z \rightarrow \imath \infty$.

The high frequency behavior of the dynamic LFC is obtained by combining the high frequency expansion of the real part of Eq.(1) for the dynamic density response with the f-sum rule and third frequency moment sum rule for the imaginary part of the dynamic density response. It reads $[1,43,104]$

$$
\begin{equation*}
G(\mathbf{q}, \infty)=I(\mathbf{q})-\frac{2 q^{2}}{m \omega_{\mathrm{p}}^{2}} T_{\mathrm{xc}} \tag{11}
\end{equation*}
$$

where $\omega_{\mathrm{p}}$ is the electron plasma frequency, $T_{\mathrm{xc}}=T-T_{0}$ is the $\mathrm{XC}$ contribution to the kinetic energy, and $I(\mathbf{q})$ is a functional of the static structure factor defined as

$$
\begin{align*}
I(q)= & \frac{1}{8 \pi^{2} n} \int_{0}^{\infty} d k k^{2}[S(k)-1]\left[\frac{5}{3}-\frac{k^{2}}{q^{2}}+\frac{\left(k^{2}-q^{2}\right)^{2}}{2 k q^{3}} \times\right. \\
& \left.\ln \left|\frac{k+q}{k-q}\right|\right] \tag{12}
\end{align*}
$$

that is sometimes referred to as the Pathak-Vashishta functional [105-107]. Combining the above, in normalized units, the imaginary-frequency LFC at the limit of infinite Matsubara order becomes

$$
\begin{equation*}
\widetilde{G}(\mathbf{x}, l \rightarrow \infty)=I(\mathbf{x})-\frac{3}{2} \pi \lambda r_{\mathrm{s}} \tau_{\mathrm{xc}} x^{2} \tag{13}
\end{equation*}
$$

where $\lambda=1 /\left(d q_{\mathrm{F}}\right)=[4 /(9 \pi)]^{1 / 3}$ for the numerical constant, $\tau_{\mathrm{xc}}$ is the XC kinetic energy in Hartree units and the Pathak-Vashishta functional is given by

$$
\begin{align*}
I(x)= & \frac{3}{8} \int_{0}^{\infty} d y y^{2}[S(y)-1]\left[\frac{5}{3}-\frac{y^{2}}{x^{2}}+\frac{\left(y^{2}-x^{2}\right)^{2}}{2 y x^{3}} \times\right. \\
& \left.\ln \left|\frac{y+x}{y-x}\right|\right] \tag{14}
\end{align*}
$$

After straightforward Taylor expansions with respect to the small $x / y$ and $y / x$ arguments, respectively, the long wavelength and short wavelength limits of the PathakVashishta functional are found to be [43]

$$
\begin{align*}
I(x \rightarrow 0) & =-\frac{1}{5} \pi \lambda r_{\mathrm{s}} v_{\mathrm{int}} x^{2}  \tag{15}\\
I(x \rightarrow \infty) & =\frac{2}{3}[1-g(0)] \tag{16}
\end{align*}
$$

where the interaction energy in Hartree units is given by $v_{\text {int }}=\left(\pi \lambda r_{\mathrm{s}}\right)^{-1} \int_{0}^{\infty} d y[S(y)-1]$ and the on-top pair correlation function by $g(0)=1+(3 / 2) \int_{0}^{\infty} d y y^{2}[S(y)-1]$. Combining the above, the long and short wavelength limits of the imaginary-frequency LFC at the limit of infinite Matsubara order become [43]

$$
\begin{align*}
\widetilde{G}(\mathbf{x} \rightarrow 0, l \rightarrow \infty) & =-\frac{1}{10} \pi \lambda r_{\mathrm{s}}\left(2 v_{\mathrm{int}}+15 \tau_{\mathrm{xc}}\right) x^{2}  \tag{17}\\
\widetilde{G}(\mathbf{x} \rightarrow \infty, l \rightarrow \infty) & =-\frac{3}{2} \pi \lambda r_{\mathrm{s}} \tau_{\mathrm{xc}} x^{2} \tag{18}
\end{align*}
$$

## D. Effective static approximation

A neural-network representation of the static limit of the fully dynamic LFC, i.e., $G(\mathbf{q}, 0)=\lim _{\omega \rightarrow 0} G(\mathbf{q}, \omega)=$ $\lim _{l \rightarrow 0} \widetilde{G}\left(\mathbf{q}, z_{l}\right)$, of the warm dense UEG is available [71]. The corresponding static approximation is obtained by setting $G(\mathbf{q}, \omega) \equiv G(\mathbf{q}, 0) \equiv G(\mathbf{q})$ in Eq. (1) and leads to an explicitly static dielectric scheme, in which the large wave number limit of the LFC is connected to the on-top pair correlation function via $[70,74]$

$$
\begin{equation*}
\lim _{q \rightarrow \infty} G(\mathbf{q})=1-g(0) \tag{19}
\end{equation*}
$$

In practice, the static limit of the fully dynamic LFC quadratically diverges for large $q[18,104,108]$, which implies an unphysical divergence of $g(0)$. To correct this error, it has been subsequently suggested to combine the exact static limit for small to intermediate wave numbers $q \lesssim 3 q_{\mathrm{F}}$ ( $q_{\mathrm{F}}$ the Fermi wave number [1]) with restricted
PIMC results for $g(0)$ [72]. The resulting effective static approximation (ESA) can be expressed as [70]

$$
\begin{equation*}
G_{\mathrm{ESA}}(\mathbf{q})=A(\mathbf{q})[1-g(0)]+G(\mathbf{q}, 0)[1-A(\mathbf{q})] \tag{20}
\end{equation*}
$$

where $A(\mathbf{q})$ is a suitable activation function for which $A(0)=0$ and $A(q \rightarrow \infty)=1$. A corresponding analytical parametrization of Eq.(20) has also been proposed [109].

## III. RESULTS

We use the extended ensemble PIMC sampling algorithm [88] as implemented in the ISHTAR code [110]. All PIMC results are freely available online [111].

## A. Dynamic density response and local field correction

We have carried out PIMC simulations of the strongly coupled electron liquid at $r_{s}=20$ [where $r_{s}=d / a_{\mathrm{B}}, d$ being the Wigner-Seitz radius and $a_{\mathrm{B}}$ the Bohr radius] and $\Theta=1$. These conditions give rise to an interesting rotontype feature in the dynamic structure factor $[5,8,9]$, which is related to an effective electron-electron attraction in the medium [9]. In addition, they constitute an interesting test bed for dielectric theories $[77,79,80]$ and have been used to illustrate the shortcomings of the static approximation $[109]$.

In Fig. 1a), we show PIMC results for the ITCF for three representative wave numbers. Since the physical meaning of its $\tau$-dependence has been discussed in the existing literature [92, 95, 112], we here restrict ourselves to a very brief summary of the main trends. First, it is easy to see that it holds that $F(\mathbf{q}, 0)=S(\mathbf{q})$. For the UEG, it also holds [113]

$$
\begin{align*}
\lim _{q \rightarrow 0} S(\mathbf{q}) & =\frac{q^{2}}{2 \omega_{p}} \operatorname{coth}\left(\frac{\beta \omega_{p}}{2}\right) \text { and }  \tag{21}\\
\lim _{q \gg q_{\mathrm{F}}} S(\mathbf{q}) & =1 \tag{22}
\end{align*}
$$

where $\omega_{p}=\sqrt{3 / r_{s}^{3}}$ is the plasma frequency, which explains the observed trends in Fig. 1a) for $\tau=0$. Second, the slope of $F(\mathbf{q}, \tau)$ around $\tau=0$ is governed by the wellknown f-sum rule [92]

$$
\begin{equation*}
\left.\frac{\partial}{\partial \tau} F(\mathbf{q}, \tau)\right|_{\tau=0}=-\frac{q^{2}}{2} \tag{23}
\end{equation*}
$$

which explains the increasingly steep $\tau$-decay for large wave numbers. This trend originates from quantum delocalization effects, as correlations eventually vanish along the imaginary-time propagation when the wave length $\lambda_{q}=2 \pi / q$ becomes comparable to the thermal wavelength $\lambda_{\beta}=\sqrt{2 \pi \beta}[92,112]$. Finally, we again note the symmetry of $F(\mathbf{q}, \tau)$ around $\tau=\beta / 2$, which is equivalent to the detailed balance relation $S(\mathbf{q},-\omega)=e^{-\beta \omega} S(\mathbf{q}, \omega)$ of the dynamic structure factor $[1,52]$.
![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-05.jpg?height=1320&width=1610&top_left_y=164&top_left_x=239)

FIG. 1. a) PIMC results for the ITCF of the UEG at $r_{s}=20$ and $\Theta=1$ for three wave numbers; b-d) $\tau$-resolved contribution to $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)[\mathrm{cf}$. Eq. (8)] for the three wave numbers and for $l=0$ (solid green), $l=1$ (dashed red), $l=2$ (dotted blue), and $l=4$ (double-dashed yellow).

In Figs. 1b)-d), we show the corresponding contributions to the dynamic Matsubara density response function $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)[\mathrm{cf}$. Eq. (8)] for these three $q$-values. Obviously, the case of $l=0$ corresponds to the ITCF itself, and Eq. (8) reverts to the usual imaginary-time version of the fluctuation-dissipation theorem, Eq. (6). With increasing Matsubara frequency order $l$, the contributions to $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ oscillate around zero, leading to cancellations. Evidently, these cancellations are reduced for larger wave numbers $q$, where the ITCF exhibits a steeper decay.

In Fig. 2, we show the full dynamic Matsubara density response function in the $q$-l-plane. The dashed black line corresponds to the static limit $\chi(\mathbf{q}, 0)$, which has been discussed extensively in the literature, e.g. [7, 71, 77, 79, 80]. For any classical system, this cutout would already completely determine the static structure factor owing to the exact connection $[114,115]$

$$
\begin{equation*}
S^{\mathrm{cl}}(\mathbf{q})=-\frac{\chi^{\mathrm{cl}}(\mathbf{q}, 0)}{n \beta} \tag{24}
\end{equation*}
$$

In other words, all contributions to Eq. (4) for $|l|>0$ are quantum mechanical in origin and would, indeed, vanish for classical point particles where the ITCF is constant with respect to $\tau$. From Fig. 2, it is evident that these quantum contributions become increasingly important with increasing $q$, i.e., on smaller length scales, as it is expected. Indeed, any system exhibits quantum mechanical behaviour on sufficiently small length scales; the large wave number limit thus always requires a full quantum mechanical description. In practice, this leads to increasingly large truncation parameters $l_{\max }$ that are required to converge the Matsubara series Eq. (4), which is a well known issue in dielectric theories $[74,79,80,116]$.

Let us next consider the difference between our quasiexact PIMC results for $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ and the RPA, which is shown in Fig. 3. Remarkably, by far the most pronounced difference occurs in the static limit of $l=0$, which explains the previously observed good performance of the static approximation; for the latter, the $l=0$ limit is, by

![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-06.jpg?height=620&width=873&top_left_y=373&top_left_x=171)

FIG. 2. The dynamic Matsubara density response $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ [cf. Eq. (8)] of the UEG at $r_{s}=20$ and $\Theta=1$, with $l$ being the integer index (order) of the Matsubara frequency.

definition, exactly reproduced, even though this comes at the expense of a small systematic error for $|l|>0$. In addition, we find that dynamic XC-effects are mostly limited to the small- $l$ regime, and the difference between PIMC and RPA decays substantially faster with the Matsubara frequency than $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ itself.

This can be seen particularly well in Fig. 4, where we show $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ as a function of $l$ for three relevant wave numbers. For $q=0.63 q_{\mathrm{F}}$, differences between our new PIMC results (black stars) and the mean-field based RPA are mainly limited to $l=0$. This changes for $q=1.88 q_{\mathrm{F}}$ and $q=3.75 q_{\mathrm{F}}$, where we find small yet significant contributions up to $l=2$ and $l=3$, respectively. In addition, we include the static approximation $G(\mathbf{q}, 0)$ and ESA as the dashed green and dotted red lines. For the two smallest depicted wave numbers, both data sets are in excellent agreement with the PIMC reference data for all $l$. For $q=3.75 q_{\mathrm{F}}$, the static approximation $G(\mathbf{q}, 0)$ reproduces the PIMC results for $l=0$ by design, whereas the ESA somewhat deviates. Interestingly, we find opposite trends for $l \geq 1$, which has consequences for the computation of the static structure factor as we explain below.

Let us next consider the dynamic Matsubara local field correction, which is shown in Fig. 5 in the relevant $q-l$ plane. We stress the remarkable stability of the inversion via Eq. (9) in this case due to the high quality (i.e., small error bars) of the PIMC results for $F(\mathbf{q}, \tau)$ and $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$. The dashed black lines show the usual static limit, $G(\mathbf{q}, 0)$, which exhibits the well known form given by the compressibility sum-rule at small $q$ [109], a maximum around twice the Fermi wave number, and again a

![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-06.jpg?height=626&width=854&top_left_y=365&top_left_x=1102)

FIG. 3. Deviations between the exact PIMC and RPA results for the dynamic Matsubara frequency response $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ of the $\mathrm{UEG}$ at $r_{s}=20$ and $\Theta=1$.

quadratic increase in the limit of large $q$ [104]. The full dynamic LFC exhibits a fairly complex behaviour that is more difficult to interpret than the density response shown in Fig. 2 above. Indeed, $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ attains negative values for large $l$, and exhibits an increasingly steep decay with $l$ for large wave numbers. Obviously, this decay is not captured by the static approximation $G(\mathbf{q}, 0)$, which is the root cause of the spurious behaviour in the static structure factor reported in Refs. [70, 109]. Overall, the dynamic Matsubara local field correction exhibits a smooth behaviour without any sharp peaks or edges, which renders its accurate parametrization with respect to $q, z_{l}$, but also $r_{s}$ and $\Theta$ a promising possibility that will be explored in dedicated future works.

In Fig. 6, we show the local field correction as a function of the wave number; the black stars correspond to $G(\mathbf{q}, 0)$, which is in excellent agreement with the neural network representation from Ref. [71], as it is expected. The ESA, on the other hand, is in perfect agreement with both data sets, but starts to deviate around $q=2.5 q_{\mathrm{F}}$. It converges towards unity for large $q$ as the on-top pair correlation function vanishes in the electron liquid regime, $g(0)=0$. The double-dashed yellow and dash-dotted blue curves show our new PIMC results for $l=1$ and $l=2$, respectively. They are systematically lower than the static limit over the entire $q$-range. The ESA curve thus does indeed constitute an effectively frequency-averaged LFC, which explains its good performance.
![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-07.jpg?height=1470&width=838&top_left_y=192&top_left_x=182)

FIG. 4. The dynamic Matsubara density response $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ [cf. Eq. (8)] of the UEG at $r_{s}=20$ and $\Theta=1$ as a function of the Matsubara order $l$ for three wave numbers $q$. Black stars: PIMC results for $N=34$; double-dashed yellow: RPA; dashed green: static approximation; dotted red: ESA.

## B. Static structure factor

In Fig. 7, we show the static structure factor $S(\mathbf{q})$ at the same conditions as in the previous sections. We note that a similar investigation has been provided in Ref. [109]. Overall, both the static approximation $G(\mathbf{q}, 0)$ (solid green) and ESA (dotted red) qualitatively reproduce the PIMC results (black crosses) over the entire $q$-range. The peak position is reproduced with high accuracy, whereas the peak height is equally overestimated by both approaches. Yet, ESA basically becomes exact for $q \gtrsim 3 q_{\mathrm{F}}$, whereas the static approximation $G(\mathbf{q}, 0)$ does not converge towards the expected limit of unity in

![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-07.jpg?height=569&width=805&top_left_y=255&top_left_x=1148)

FIG. 5. PIMC results for the dynamic Matsubara local field correction $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ [cf. Eq. (9)] of the UEG at $r_{s}=20$ and $\Theta=1$, with $l$ the Matsubara order.

![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-07.jpg?height=669&width=835&top_left_y=1064&top_left_x=1098)

FIG. 6. Dynamic Matsubara local field correction $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ of the UEG as a function of the wave number $q$ for different Matsubara orders at $r_{s}=20$ and $\Theta=1$. The solid green line has been obtained using the neural network representation from Ref. [71] and the dotted red line has been obtained from the analytical ESA parametrization presented in Ref. [109].

the depicted range of wave numbers.

To trace this trend to the impact of dynamic XCeffects, we define the deviation measure

$$
\begin{equation*}
\Delta \mathrm{XC}\left(\mathbf{q}, z_{l}\right)=-\frac{1}{n \beta}\left[\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)-\widetilde{\chi}_{\mathrm{RPA}}\left(\mathbf{q}, z_{l}\right)\right] \tag{25}
\end{equation*}
$$

which corrects Eq. (4) with respect to the mean-field based RPA. In Fig. 8, we show the dependence of Eq. (25) on $l$ for two representative wave numbers. For $q=1.25 q_{\mathrm{F}}$ (top panel), both the static approximation $G(\mathbf{q}, 0)$ and

![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-08.jpg?height=908&width=829&top_left_y=197&top_left_x=190)

FIG. 7. Static structure factor $S(\mathbf{q})$ of the UEG at $r_{s}=20$ and $\Theta=1$. Black crosses: PIMC results for $N=34$; doubledashed yellow: RPA; solid green: static approximation; dotted red: ESA. The inset shows a magnified segment around the peak.

the ESA reproduce the PIMC reference data equally well, and their corresponding result for $S(\mathbf{q})$ is highly accurate. For $q=3.75 q_{\mathrm{F}}$, we find an error cancellation in the ESA; in contrast, the static approximation $G(\mathbf{q}, 0)$ is exact for $l=0$, but significantly overestimates the true XC-correction for $1 \leq l \lesssim 7$. The accumulation of these terms then leads to the observed overestimation of the static structure factor for large $q$.

## C. Asymptotic high frequency behavior of the Matsubara local field correction

Let us conclude by analyzing the high-frequency limit of the dynamic Matsubara local field correction $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$. In Fig. 9, we plot $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ for various $l$ in the range of $0 \leq l \leq 13$. It is evident that the data become increasingly noisy for large $l$, which is a consequence of the diminishing impact of the local field correction onto $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$. For every constant wave number, we observe a monotonic decrease of $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ from the static limit towards the asymptotic high-frequency limit, Eq. (11), that is shown as the bold solid red curve. In practice, the convergence of $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ towards $\widetilde{G}(\mathbf{q}, \infty)$ with $l$ becomes substantially slower with increasing wavenumber due to the more pronounced importance of quantum delocalization effects, as

![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-08.jpg?height=1326&width=838&top_left_y=188&top_left_x=1099)

FIG. 8. Dynamic XC-effect metric [cf. Eq. (25)] as a function of the Matsubara order $l$ for two representative wave numbers q. Solid black: PIMC; dashed green: static approximation; dotted red: ESA.

it has already been explained above. Overall, Fig. 9 suggests that, at least for UEG state points for which the XC contribution to the kinetic energy is positive $\left(\tau_{\mathrm{xc}}>0\right)$, the dynamic Matsubara local field correction $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ is upper bounded by its static limit $\widetilde{G}(\mathbf{q}, 0)$ and lower bounded by its high frequency limit $\widetilde{G}(\mathbf{q}, \infty)$.

In Fig. 10, we decompose the individual contributions to $\widetilde{G}(\mathbf{q}, \infty)$. The dotted green line corresponds to the kinetic energy contribution, which is parabolic for all $q$. The dash-dotted blue curve shows the Pathak-Vashishta functional [Eq. (12)] that depends on the static structure factor $S(\mathbf{q})$. It converges towards $2 / 3$ in the $q \rightarrow \infty$ limit as the on-top pair correlation function vanishes at these parameters [Eq. (16)], and attains a parabola in the limit of $q \rightarrow 0$ [Eq. (15)]; both limits are shown as the dashed light grey curves in Fig. 10 and are in excellent agreement with our numerical results for $I(\mathbf{q})$. Naturally, the same holds for the dashed black lines that depict the

![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-09.jpg?height=1288&width=848&top_left_y=191&top_left_x=172)

FIG. 9. Matsubara order ( $l$ ) resolved wave number dependence of the dynamic local field correction $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ of the UEG at $r_{s}=20$ and $\Theta=1$. Dotted green: neural network representation from Ref. [71]; dotted red: ESA parametrization from Ref. [109]; solid red: high frequency limit, Eq. (11), using $T_{\mathrm{xc}}$ computed from the XC-free energy parametrization by Groth et al. [32] (GDSMFB).

corresponding limits of $\widetilde{G}(\mathbf{q}, \infty)$.

The above two findings clearly hint at the intriguing possibility of constructing an analytic four-parameter representation of the full dynamic Matsubara local field correction $\widetilde{G}\left(\mathbf{q}, z_{l} ; r_{s}, \Theta\right)$ over a broad range of UEG parameters.

## IV. SUMMARY AND DISCUSSION

In this work, we have presented an analysis of dynamic XC-effects in the strongly coupled electron liquid. This has been achieved on the basis of highly accurate direct PIMC results for the Matsubara density response function $\widetilde{\chi}\left(\mathbf{q}, z_{l}\right)$ that have been obtained from the ITCF

![](https://cdn.mathpix.com/cropped/2024_06_04_5a6e1d4c1e2343483034g-09.jpg?height=673&width=832&top_left_y=192&top_left_x=1102)

FIG. 10. Wave number dependence of the high frequency limit $(l \rightarrow \infty)$ of the dynamic Matsubara local field correction $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$ of the UEG at $r_{s}=20$ and $\Theta=1$. Solid red: Full $G(\mathbf{q}, \infty)$, Eq. (11); dash-dotted blue: $I(\mathbf{q})$, Eq. (12); dashed green: $T_{\mathrm{xc}}$ term evaluated using $T_{\mathrm{xc}}$ computed from the XCfree energy parametrization by Groth et al. [32]. The dashed grey and black lines show the asymptotic short and long wavelength limits of $I(\mathbf{q})$ and $G(\mathbf{q}, \infty)$, respectively, cf. Sec. II C.

$F(\mathbf{q}, \tau)$ using the recently derived Fourier-Matsubara expansion by Tolias et al. [84]. In particular, this approach allows us to obtain the full dynamic Matsubara local field correction $\widetilde{G}\left(\mathbf{q}, z_{l}\right)$, which exhibits a nontrivial behaviour. Our results provide new insights into the complex interplay between XC-correlation and quantum delocalization effects, and explain the observed differences between the static approximation and ESA at large wave numbers $[70,109]$.

We are convinced that the presented methodology opens up a number of possibilities for impactful future research. First, the dynamic LFC constitutes the key property in dielectric theories $[64,78,80,83]$; the availability of highly accurate PIMC results can thus give new insights into existing approximations, and guide the development of new approaches. Second, extensive future results for the dynamic LFC might be used as input to construct a parametrization $\widetilde{G}\left(\mathbf{q}, z_{l} ; r_{s}, \Theta\right)$ that covers (substantial parts of) the WDM regime.

Such a parametrization would be key input for the construction of advanced, non-local XC-functionals for thermal DFT simulations based on the adiabatic-connection formula and the fluctuation-dissipation theorem [58]. Finally, we note that direct PIMC simulations that retain access to the full imaginary-time structure have recently been presented for warm dense hydrogen [95, 117] and beryllium $[99,117]$, which opens up the enticing possibility to study species-resolved dynamic XC-effects in real materials.

## ACKNOWLEDGMENTS

This work was partially supported by the Center for Advanced Systems Understanding (CASUS), financed by Germany's Federal Ministry of Education and Research (BMBF) and the Saxon state government out of the State budget approved by the Saxon State Parliament. Further support is acknowledged for the CASUS Open Project Guiding dielectric theories with ab initio quantum Monte Carlo simulations: from the strongly coupled electron liquid to warm dense matter. This work has received funding from the European Research Council (ERC) under the European Union's Horizon 2022 research and innovation programme (Grant agreement No.
101076233, "PREXTREME"). Views and opinions expressed are however those of the authors only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them. Computations were performed on a Bull Cluster at the Center for Information Services and High-Performance Computing (ZIH) at Technische Universität Dresden, at the Norddeutscher Verbund für Hoch- und Höchstleistungsrechnen (HLRN) under grant mvp00024, and on the HoreKa supercomputer funded by the Ministry of Science, Research and the Arts BadenWürttemberg and by the Federal Ministry of Education and Research.
[1] G. Giuliani and G. Vignale, Quantum Theory of the Electron Liquid (Cambridge University Press, Cambridge, 2008).

[2] P.-F. Loos and P. M. W. Gill, "The uniform electron gas," Comput. Mol. Sci 6, 410-429 (2016).

[3] T. Dornheim, S. Groth, and M. Bonitz, "The uniform electron gas at warm dense matter conditions," Phys. Reports 744, 1-86 (2018).

[4] G.D. Mahan, Many-Particle Physics, Physics of Solids and Liquids (Springer US, 1990).

[5] Yasutami Takada, "Emergence of an excitonic collective mode in the dilute electron gas," Phys. Rev. B 94, 245106 (2016).

[6] T. Dornheim, S. Groth, J. Vorberger, and M. Bonitz, "Ab initio path integral Monte Carlo results for the dynamic structure factor of correlated electrons: From the electron liquid to warm dense matter," Phys. Rev. Lett. 121, 255001 (2018).

[7] S. Groth, T. Dornheim, and J. Vorberger, "Ab initio path integral Monte Carlo approach to the static and dynamic density response of the uniform electron gas," Phys. Rev. B 99, 235122 (2019).

[8] Tobias Dornheim, Zhandos Moldabekov, Jan Vorberger, Hanno Kählert, and Michael Bonitz, "Electronic pair alignment and roton feature in the warm dense electron gas," Communications Physics 5, 304 (2022).

[9] Tobias Dornheim, Panagiotis Tolias, Zhandos A. Moldabekov, Attila Cangi, and Jan Vorberger, "Effective electronic forces and potentials from ab initio path integral Monte Carlo simulations," The Journal of Chemical Physics 156, 244113 (2022).

[10] Jaakko Koskelo, Lucia Reining, and Matteo Gatti, "Short-range excitonic phenomena in low-density metals," (2023), arXiv:2301.00474 [cond-mat.str-el].

[11] E. Wigner, "On the interaction of electrons in metals," Phys. Rev. 46, 1002-1011 (1934).

[12] N. D. Drummond, Z. Radnai, J. R. Trail, M. D. Towler, and R. J. Needs, "Diffusion quantum Monte Carlo study of three-dimensional Wigner crystals," Phys. Rev. B 69, 085116 (2004).

[13] Sam Azadi and N. D. Drummond, "Low-density phase diagram of the three-dimensional electron gas," Phys. Rev. B 105, 245135 (2022).
[14] S. H. Vosko, L. Wilk, and M. Nusair, "Accurate spindependent electron liquid correlation energies for local spin density calculations: a critical analysis," Canadian Journal of Physics 58, 1200-1211 (1980).

[15] John P. Perdew and Yue Wang, "Accurate and simple analytic representation of the electron-gas correlation energy," Phys. Rev. B 45, 13244-13249 (1992).

[16] J. P. Perdew and Alex Zunger, "Self-interaction correction to density-functional approximations for manyelectron systems," Phys. Rev. B 23, 5048-5079 (1981).

[17] Paola Gori-Giorgi, Francesco Sacchetti, and Giovanni B. Bachelet, "Analytic static structure factors and pair-correlation functions for the unpolarized homogeneous electron gas," Phys. Rev. B 61, 7353-7363 (2000).

[18] M. Corradini, R. Del Sole, G. Onida, and M. Palummo, "Analytical expressions for the local-field factor $g(q)$ and the exchange-correlation kernel $K_{\text {xc }}(r)$ of the homogeneous electron gas," Phys. Rev. B 57, 14569 (1998).

[19] D. M. Ceperley and B. J. Alder, "Ground state of the electron gas by a stochastic method," Phys. Rev. Lett. 45, 566-569 (1980).

[20] G. G. Spink, R. J. Needs, and N. D. Drummond, "Quantum Monte Carlo study of the three-dimensional spin-polarized homogeneous electron gas," Phys. Rev. B 88, 085121 (2013).

[21] S. Moroni, D. M. Ceperley, and G. Senatore, "Static response from quantum Monte Carlo calculations," Phys. Rev. Lett 69, 1837 (1992).

[22] S. Moroni, D. M. Ceperley, and G. Senatore, "Static response and local field factor of the electron gas," Phys. Rev. Lett 75, 689 (1995).

[23] R. O. Jones, "Density functional theory: Its origins, rise to prominence, and future," Rev. Mod. Phys. 87, 897923 (2015).

[24] R. Betti and O. A. Hurricane, "Inertial-confinement fusion with lasers," Nature Physics 12, 435-448 (2016).

[25] O. A. Hurricane, P. K. Patel, R. Betti, D. H. Froula, S. P. Regan, S. A. Slutz, M. R. Gomez, and M. A. Sweeney, "Physics principles of inertial confinement fusion and u.s. program overview," Rev. Mod. Phys. 95, 025005 (2023).

[26] Abu-Shawareb et al. (The Indirect Drive ICF Collaboration), "Achievement of target gain larger than unity in an inertial fusion experiment," Phys. Rev. Lett. 132,

## 065102 (2024).

[27] Alessandra Benuzzi-Mounaix, Stéphane Mazevet, Alessandra Ravasio, Tommaso Vinci, Adrien Denoeud, Michel Koenig, Nourou Amadou, Erik Brambrink, Floriane Festa, Anna Levy, Marion Harmand, Stéphanie Brygoo, Gael Huser, Vanina Recoules, Johan Bouchet, Guillaume Morard, François Guyot, Thibaut de Resseguier, Kohei Myanishi, Norimasa Ozaki, Fabien Dorchies, Jerôme Gaudin, Pierre Marie Leguay, Olivier Peyrusse, Olivier Henry, Didier Raffestin, Sebastien Le Pape, Ray Smith, and Riccardo Musella, "Progress in warm dense matter study with applications to planetology," Phys. Scripta T161, 014060 (2014).

[28] A. Becker, W. Lorenzen, J. J. Fortney, N. Nettelmann, M. Schöttler, and R. Redmer, "Ab initio equations of state for hydrogen (h-reos.3) and helium (he-reos.3) and their implications for the interior of brown dwarfs," Astrophys. J. Suppl. Ser 215, 21 (2014).

[29] Andrea L. Kritcher, Damian C. Swift, Tilo Döppner, Benjamin Bachmann, Lorin X. Benedict, Gilbert W. Collins, Jonathan L. DuBois, Fred Elsner, Gilles Fontaine, Jim A. Gaffney, Sebastien Hamel, Amy Lazicki, Walter R. Johnson, Natalie Kostinski, Dominik Kraus, Michael J. MacDonald, Brian Maddox, Madison E. Martin, Paul Neumayer, Abbas Nikroo, Joseph Nilsen, Bruce A. Remington, Didier Saumon, Phillip A. Sterne, Wendi Sweet, Alfredo A. Correa, Heather D. Whitley, Roger W. Falcone, and Siegfried H. Glenzer, "A measurement of the equation of state of carbon envelopes of white dwarfs," Nature 584, 51-54 (2020).

[30] Fionn D. Malone, N. S. Blunt, Ethan W. Brown, D. K. K. Lee, J. S. Spencer, W. M. C. Foulkes, and James J. Shepherd, "Accurate exchange-correlation energies for the warm dense electron gas," Phys. Rev. Lett. 117, 115701 (2016)

[31] T. Dornheim, S. Groth, T. Sjostrom, F. D. Malone, W. M. C. Foulkes, and M. Bonitz, "Ab initio quantum Monte Carlo simulation of the warm dense electron gas in the thermodynamic limit," Phys. Rev. Lett. 117, 156403 (2016).

[32] S. Groth, T. Dornheim, T. Sjostrom, F. D. Malone, W. M. C. Foulkes, and M. Bonitz, "Ab initio exchangecorrelation free energy of the uniform electron gas at warm dense matter conditions," Phys. Rev. Lett. 119, 135001 (2017).

[33] Valentin V. Karasiev, Travis Sjostrom, James Dufty, and S. B. Trickey, "Accurate homogeneous electron gas exchange-correlation free energy for local spin-density calculations," Phys. Rev. Lett. 112, 076403 (2014).

[34] Valentin V. Karasiev, S. B. Trickey, and James W. Dufty, "Status of free-energy representations for the homogeneous electron gas," Phys. Rev. B 99, 195134 (2019).

[35] T. Dornheim, S. Groth, J. Vorberger, and M. Bonitz, "Permutation blocking path integral Monte Carlo approach to the static density response of the warm dense electron gas," Phys. Rev. E 96, 023203 (2017).

[36] S. Groth, T. Dornheim, and M. Bonitz, "Configuration path integral Monte Carlo approach to the static density response of the warm dense electron gas," J. Chem. Phys 147, 164108 (2017)

[37] M. Bonitz, T. Dornheim, Zh. A. Moldabekov, S. Zhang, P. Hamann, H. Kählert, A. Filinov, K. Ramakrishna, and J. Vorberger, "Ab initio simulation of warm dense matter," Physics of Plasmas 27, 042710 (2020).

[38] F. Graziani, M. P. Desjarlais, R. Redmer, and S. B. Trickey, eds., Frontiers and Challenges in Warm Dense Matter (Springer, International Publishing, 2014).

[39] D. M. Ceperley, "Path integrals in the theory of condensed helium," Rev. Mod. Phys 67, 279 (1995).

[40] M. F. Herman, E. J. Bruskin, and B. J. Berne, "On path integral Monte Carlo simulations," The Journal of Chemical Physics 76, 5150-5155 (1982).

[41] Torben Ott, Hauke Thomsen, Jan Willem Abraham, Tobias Dornheim, and Michael Bonitz, "Recent progress in the theory and simulation of strongly correlated plasmas: phase transitions, transport, quantum, and magnetic field effects," The European Physical Journal D 72, 84 (2018).

[42] Tobias Dornheim, Zhandos A. Moldabekov, Kushal Ramakrishna, Panagiotis Tolias, Andrew D. Baczewski, Dominik Kraus, Thomas R. Preston, David A. Chapman, Maximilian P. Böhme, Tilo Döppner, Frank Graziani, Michael Bonitz, Attila Cangi, and Jan Vorberger, "Electronic density response of warm dense matter," Physics of Plasmas 30, 032705 (2023).

[43] A. A. Kugler, "Theory of the local field correction in an electron gas," J. Stat. Phys 12, 35 (1975).

[44] Xiaolei Zan, Chengliang Lin, Yong Hou, and Jianmin Yuan, "Local field correction to ionization potential depression of ions in warm or hot dense matter," Phys. Rev. E 104, 025203 (2021).

[45] Zh. A. Moldabekov, T. Dornheim, M. Bonitz, and T. S. Ramazanov, "Ion energy-loss characteristics and friction in a free-electron gas at warm dense matter and nonideal dense plasma conditions," Phys. Rev. E 101, 053203 (2020).

[46] Carsten Fortmann, August Wierling, and Gerd Röpke, "Influence of local-field corrections on thomson scattering in collision-dominated two-component plasmas," Phys. Rev. E 81, 026405 (2010).

[47] Carsten A. Ullrich, Time-Dependent Density-Functional Theory: Concepts and Applications (Oxford University Press, 2011).

[48] Zhandos A. Moldabekov, Michele Pavanello, Maximilian P. Böhme, Jan Vorberger, and Tobias Dornheim, "Linear-response time-dependent density functional theory approach to warm dense matter with adiabatic exchange-correlation kernels," Phys. Rev. Res. 5, 023089 (2023).

[49] Zhandos Moldabekov, Maximilian Böhme, Jan Vorberger, David Blaschke, and Tobias Dornheim, "Ab initio static exchange-correlation kernel across jacob's ladder without functional derivatives," Journal of Chemical Theory and Computation 19, 1286-1299 (2023).

[50] S. H. Glenzer and R. Redmer, "X-ray thomson scattering in high energy density plasmas," Rev. Mod. Phys 81, 1625 (2009).

[51] T. Döppner, M. Bethkenhagen, D. Kraus, P. Neumayer, D. A. Chapman, B. Bachmann, R. A. Baggott, M. P. Böhme, L. Divol, R. W. Falcone, L. B. Fletcher, O. L. Landen, M. J. MacDonald, A. M. Saunders, M. Schörner, P. A. Sterne, J. Vorberger, B. B. L. Witte, A. Yi, R. Redmer, S. H. Glenzer, and D. O. Gericke, "Observing the onset of pressure-driven k-shell delocalization," Nature 618, 270-275 (2023).

[52] Tobias Dornheim, Maximilian Böhme, Dominik Kraus, Tilo Döppner, Thomas R. Preston, Zhandos A. Mold-
abekov, and Jan Vorberger, "Accurate temperature diagnostics for matter under extreme conditions," Nature Communications 13, 7911 (2022).

[53] Tobias Dornheim, Maximilian P. Böhme, David A. Chapman, Dominik Kraus, Thomas R. Preston, Zhandos A. Moldabekov, Niclas Schlünzen, Attila Cangi, Tilo Döppner, and Jan Vorberger, "Imaginary-time correlation function thermometry: A new, high-accuracy and model-free temperature analysis technique for $\mathrm{x}$ ray Thomson scattering data," Physics of Plasmas 30, 042707 (2023).

[54] S. Frydrych, J. Vorberger, N. J. Hartley, A. K. Schuster, K. Ramakrishna, A. M. Saunders, T. van Driel, R. W. Falcone, L. B. Fletcher, E. Galtier, E. J. Gamboa, S. H. Glenzer, E. Granados, M. J. MacDonald, A. J. MacKinnon, E. E. McBride, I. Nam, P. Neumayer, A. Pak, K. Voigt, M. Roth, P. Sun, D. O. Gericke, T. Döppner, and D. Kraus, "Demonstration of x-ray thomson scattering as diagnostics for miscibility in warm dense matter," Nature Communications 11, 2620 (2020).

[55] D. Kraus, B. Bachmann, B. Barbrel, R. W. Falcone, L. B. Fletcher, S. Frydrych, E. J. Gamboa, M. Gauthier, D. O. Gericke, S. H. Glenzer, S. Göde, E. Granados, N. J. Hartley, J. Helfrich, H. J. Lee, B. Nagler, A. Ravasio, W. Schumaker, J. Vorberger, and T. Döppner, "Characterizing the ionization potential depression in dense carbon plasmas with high-precision spectrally resolved x-ray scattering," Plasma Phys. Control Fusion 61, 014015 (2019).

[56] Maximilian P. Böhme, Luke B. Fletcher, Tilo Döppner, Dominik Kraus, Andrew D. Baczewski, Thomas R. Preston, Michael J. MacDonald, Frank R. Graziani, Zhandos A. Moldabekov, Jan Vorberger, and Tobias Dornheim, "Evidence of free-bound transitions in warm dense matter and their impact on equation-of-state measurements," (2023), arXiv:2306.17653 [physics.plasm$\mathrm{ph}$.

[57] G. Gregori, S. H. Glenzer, W. Rozmus, R. W. Lee, and O. L. Landen, "Theoretical model of x-ray scattering as a dense matter probe," Phys. Rev. E 67, 026412 (2003).

[58] A. Pribram-Jones, P. E. Grabowski, and K. Burke, "Thermal density functional theory: Time-dependent linear response and approximate functionals from the fluctuation-dissipation theorem," Phys. Rev. Lett 116, 233001 (2016).

[59] E. K. U. Gross and W. Kohn, "Local density-functional theory of frequency-dependent linear response," Phys. Rev. Lett 55, 2850 (1985).

[60] Lucian A. Constantin and J. M. Pitarke, "Simple dynamic exchange-correlation kernel of a uniform electron gas," Phys. Rev. B 75, 245127 (2007).

[61] Zhixin Qian and Giovanni Vignale, "Dynamical exchange-correlation potentials for an electron liquid," Phys. Rev. B 65, 235121 (2002).

[62] Bogdan Dabrowski, "Dynamical local-field factor in the response function of an electron gas," Phys. Rev. B 34, 4989-4995 (1986).

[63] Adrienn Ruzsinszky, Niraj K. Nepal, J. M. Pitarke, and John P. Perdew, "Constraint-based wave vector and frequency dependent exchange-correlation kernel of the uniform electron gas," Phys. Rev. B 101, 245135 (2020).

[64] A. Holas and S. Rahman, "Dynamic local-field factor of an electron liquid in the quantum versions of the Singwi-Tosi-Land-Sjölander and Vashishta-Singwi the- ories," Phys. Rev. B 35, 2720 (1987).

[65] Tobias Dornheim and Jan Vorberger, "Finite-size effects in the reconstruction of dynamic properties from ab initio path integral Monte Carlo simulations," Phys. Rev. E 102, 063301 (2020).

[66] Paul Hamann, Tobias Dornheim, Jan Vorberger, Zhandos A. Moldabekov, and Michael Bonitz, "Dynamic properties of the warm dense electron gas based on abinitio path integral monte carlo simulations," Phys. Rev. B 102, 125150 (2020).

[67] Mark Jarrell and J.E. Gubernatis, "Bayesian inference and the analytic continuation of imaginary-time quantum Monte Carlo data," Physics Reports 269, 133-195 (1996).

[68] James P. F. LeBlanc, Kun Chen, Kristjan Haule, Nikolay V. Prokof'ev, and Igor S. Tupitsyn, "Dynamic response of an electron gas: Towards the exact exchange-correlation kernel," Phys. Rev. Lett. 129, 246401 (2022).

[69] Paul Hamann, Jan Vorberger, Tobias Dornheim, Zhandos A. Moldabekov, and Michael Bonitz, "Ab initio results for the plasmon dispersion and damping of the warm dense electron gas," Contributions to Plasma Physics 60, e202000147 (2020).

[70] Tobias Dornheim, Attila Cangi, Kushal Ramakrishna, Maximilian Böhme, Shigenori Tanaka, and Jan Vorberger, "Effective static approximation: A fast and reliable tool for warm-dense matter theory," Phys. Rev Lett. 125, 235001 (2020).

[71] T. Dornheim, J. Vorberger, S. Groth, N. Hoffmann, Zh.A. Moldabekov, and M. Bonitz, "The static local field correction of the warm dense electron gas: An ab initio path integral Monte Carlo study and machine learning representation," J. Chem. Phys 151, 194104 (2019).

[72] Ethan W. Brown, Bryan K. Clark, Jonathan L. DuBois, and David M. Ceperley, "Path-Integral Monte Carlo Simulation of the Warm Dense Homogeneous Electron Gas," Phys. Rev. Lett. 110, 146405 (2013).

[73] Setsuo Ichimaru, Hiroshi Iyetomi, and Shigenori Tanaka, "Statistical physics of dense plasmas: Thermodynamics, transport coefficients and dynamic correlations," Physics Reports 149, 91-205 (1987).

[74] S. Tanaka and S. Ichimaru, "Thermodynamics and correlational properties of finite-temperature electron liquids in the Singwi-Tosi-Land-Sjölander approximation," J. Phys. Soc. Jpn 55, 2278-2289 (1986).

[75] K. S. Singwi, M. P. Tosi, R. H. Land, and A. Sjölander, "Electron correlations at metallic densities," Phys. Rev $\mathbf{1 7 6}, 589$ (1968).

[76] T. Sjostrom and J. Dufty, "Uniform electron gas at finite temperatures," Phys. Rev. B 88, 115123 (2013).

[77] Tobias Dornheim, Travis Sjostrom, Shigenori Tanaka, and Jan Vorberger, "Strongly coupled electron liquid: Ab initio path integral Monte Carlo simulations and dielectric theories," Phys. Rev. B 101, 045129 (2020).

[78] Panagiotis Tolias, Federico Lucco Castello, Fotios Kalkavouras, and Tobias Dornheim, "Revisiting the Vashishta-Singwi dielectric scheme for the warm dense uniform electron fluid," Phys. Rev. B 109, 125134 (2024).

[79] P. Tolias, F. Lucco Castello, and T. Dornheim, "Integral equation theory based dielectric scheme for strongly coupled electron liquids," The Journal of Chemical

Physics 155, 134115 (2021).

[80] Panagiotis Tolias, Federico Lucco Castello, and Tobias Dornheim, "Quantum version of the integral equation theory-based dielectric scheme for strongly coupled electron liquids," The Journal of Chemical Physics 158, 141102 (2023).

[81] S. Tanaka, "Correlational and thermodynamic properties of finite-temperature electron liquids in the hypernetted-chain approximation," J. Chem. Phys 145, 214104 (2016).

[82] Shigenori Tanaka, "Improved equation of state for finitetemperature spin-polarized electron liquids on the basis of singwi-tosi-land-sjölander approximation," Contributions to Plasma Physics 57, 126-136 (2017).

[83] P. Arora, K. Kumar, and R. K. Moudgil, "Spin-resolved correlations in the warm-dense homogeneous electron gas," Eur. Phys. J. B 90, 76 (2017).

[84] Panagiotis Tolias, Fotios Kalkavouras, and Tobias Dornheim, "Fourier-Matsubara series expansion for imaginary-time correlation functions," J. Chem. Phys. 160, 181102 (2024).

[85] H. Kleinert, Path Integrals in Quantum Mechanics, Statistics, Polymer Physics, and Financial Markets, EBL-Schweitzer (World Scientific, 2009).

[86] M. Boninsegni, N. V. Prokofev, and B. V. Svistunov, "Worm algorithm and diagrammatic Monte Carlo: A new approach to continuous-space path integral Monte Carlo simulations," Phys. Rev. E 74, 036701 (2006).

[87] M. Boninsegni, N. V. Prokofev, and B. V. Svistunov, "Worm algorithm for continuous-space path integral Monte Carlo simulations," Phys. Rev. Lett 96, 070601 (2006).

[88] Tobias Dornheim, Maximilian Böhme, Burkhard Militzer, and Jan Vorberger, "Ab initio path integral monte carlo approach to the momentum distribution of the uniform electron gas at finite temperature without fixed nodes," Phys. Rev. B 103, 205142 (2021).

[89] Tobias Dornheim, Zhandos A. Moldabekov, and Jan Vorberger, "Nonlinear density response from imaginarytime correlation functions: Ab initio path integral Monte Carlo simulations of the warm dense electron gas," The Journal of Chemical Physics 155, 054110 (2021).

[90] B. Militzer, E.L. Pollock, and D.M. Ceperley, "Path integral monte carlo calculation of the momentum distribution of the homogeneous electron gas at finite temperature," High Energy Density Physics 30, 13-20 (2019).

[91] Tobias Dornheim, Jan Vorberger, Burkhard Militzer, and Zhandos A. Moldabekov, "Momentum distribution of the uniform electron gas at finite temperature: Effects of spin polarization," Phys. Rev. E 104, 055206 (2021).

[92] Tobias Dornheim, Zhandos Moldabekov, Panagiotis Tolias, Maximilian Böhme, and Jan Vorberger, "Physical insights from imaginary-time density-density correlation functions," Matter and Radiation at Extremes 8, 056601 (2023).

[93] Tobias Dornheim, Zhandos A Moldabekov, Jan Vorberger, and Simon Groth, "Ab initio path integral monte carlo simulation of the uniform electron gas in the high energy density regime," Plasma Physics and Controlled Fusion 62, 075003 (2020).

[94] Tobias Dornheim, Jan Vorberger, Zhandos Moldabekov, Gerd Röpke, and Wolf-Dietrich Kraeft, "The uniform electron gas at high temperatures: ab initio path integral Monte Carlo simulations and analytical theory," High
Energy Density Physics 45, 101015 (2022).

[95] Tobias Dornheim, Sebastian Schwalbe, Panagiotis Tolias, Maximilan Böhme, Zhandos Moldabekov, and Jan Vorberger, "Ab initio density response and local field factor of warm dense hydrogen," (2024), arXiv:2403.08570 [physics.plasm-ph].

[96] Tobias Dornheim, Jan Vorberger, Zhandos A. Moldabekov, and Michael Bonitz, "Nonlinear interaction of external perturbations in warm dense matter," Contributions to Plasma Physics n/a, e202100247 (2022).

[97] Tobias Dornheim, Tilo Döppner, Andrew D. Baczewski, Panagiotis Tolias, Maximilian P. Böhme, Zhandos A. Moldabekov, Divyanshu Ranjan, David A. Chapman, Michael J. MacDonald, Thomas R. Preston, Dominik Kraus, and Jan Vorberger, "X-ray Thomson scattering absolute intensity from the f-sum rule in the imaginary-time domain," arXiv (2023), 2305.15305 [physics.plasm-ph].

[98] Maximilian Schörner, Mandy Bethkenhagen, Tilo Döppner, Dominik Kraus, Luke B. Fletcher, Siegfried H. Glenzer, and Ronald Redmer, "X-ray thomson scattering spectra from density functional theory molecular dynamics simulations based on a modified chihara formula," Phys. Rev. E 107, 065207 (2023).

[99] Tobias Dornheim, Tilo Döppner, Panagiotis Tolias, Maximilian Böhme, Luke Fletcher, Thomas Gawne, Frank Graziani, Dominik Kraus, Michael MacDonald, Zhandos Moldabekov, Sebastian Schwalbe, Dirk Gericke, and Jan Vorberger, "Unraveling electronic correlations in warm dense quantum plasmas," (2024), arXiv:2402.19113 [physics.plasm-ph].

[100] Jan Vorberger, Thomas R. Preston, Nikita Medvedev, Maximilian P. Böhme, Zhandos A. Moldabekov, Dominik Kraus, and Tobias Dornheim, "Revealing nonequilibrium and relaxation in laser heated matter," Physics Letters A 499, 129362 (2024).

[101] Tobias Dornheim, Damar C. Wicaksono, Juan E. Suarez-Cardona, Panagiotis Tolias, Maximilian P. Böhme, Zhandos A. Moldabekov, Michael Hecht, and Jan Vorberger, "Extraction of the frequency moments of spectral densities from imaginary-time correlation function data," Phys. Rev. B 107, 155148 (2023).

[102] D. M. Ceperley, "Fermion nodes," Journal of Statistical Physics 63, 1237-1267 (1991).

[103] T. Dornheim, "Fermion sign problem in path integral Monte Carlo simulations: Quantum dots, ultracold atoms, and warm dense matter," Phys. Rev. E 100, 023307 (2019).

[104] A. Holas, "Exact asymptotic expression for the static dielectric function of a uniform electron liquid at large wave vector," in Strongly Coupled Plasma Physics, edited by F.J. Rogers and H.E. DeWitt (Plenum, New York, 1987).

[105] K. N. Pathak and P. Vashishta, "Electron correlations and moment sum rules," Phys. Rev. B 7, 3649-3656 (1973).

[106] Göran Niklasson, "Dielectric function of the uniform electron gas for large frequencies or wave vectors," Phys. Rev. B 10, 3052-3061 (1974).

[107] K. S. Singwi and M. P. Tosi, "Correlations in electron liquids," Solid State Physics 36, 177-266 (1981).

[108] Peng-Cheng Hou, Bao-Zong Wang, Kristjan Haule, Youjin Deng, and Kun Chen, "Exchange-correlation effect in the charge response of a warm dense electron
gas," Phys. Rev. B 106, L081126 (2022).

[109] Tobias Dornheim, Zhandos A. Moldabekov, and Panagiotis Tolias, "Analytical representation of the local field correction of the uniform electron gas within the effective static approximation," Phys. Rev. B 103, 165102 (2021).

[110] Tobias Dornheim, Maximilian Böhme, and Sebastian Schwalbe, "ISHTAR - Imaginary-time Stochastic Highperformance Tool for Ab initio Research," (2024).

[111] A link to a repository containing all PIMC raw data will be made available upon publication.

[112] Tobias Dornheim, Jan Vorberger, Zhandos A. Moldabekov, and Maximilian Böhme, "Analysing the dynamic structure of warm dense matter in the imaginarytime domain: theoretical models and simulations," Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences 381, 20220217 (2023).

[113] A. A. Kugler, "Bounds for some equilibrium properties of an electron gas," Phys. Rev. A 1, 1688 (1970).
[114] A. A. Kugler, "Collective modes, damping, and the scattering function in classical liquids," J. Stat. Phys. 8, 107-153 (1973).

[115] P. Tolias and F. Lucco Castello, "Description of longitudinal modes in moderately coupled Yukawa systems with the static local field correction," Phys. Plasmas 28, 034502 (2021).

[116] Shigenori Tanaka, Shinichi Mitake, and Setsuo Ichimaru, "Parametrized equation of state for electron liquids in the Singwi-Tosi-Land-Sjölander approximation," Phys. Rev. A 32, 1896 (1985).

[117] Tobias Dornheim, Sebastian Schwalbe, Maximilian Böhme, Zhandos Moldabekov, Jan Vorberger, and Panagiotis Tolias, "Ab initio path integral monte carlo simulations of warm dense two-component systems without fixed nodes: structural properties," J. Chem. Phys. 160, 164111 (2024).


[^0]:    * t.dornheim@hzdr.de

</end of paper 4>


