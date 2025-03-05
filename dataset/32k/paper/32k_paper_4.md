<paper 0>
# Minimod: A Finite Difference solver for Seismic Modeling 

Jie Meng* Andreas Atle ${ }^{*} \quad$ Henri Calandra ${ }^{\dagger} \quad$ Mauricio Araya-Polo*

July 14,2020


#### Abstract

This article introduces a benchmark application for seismic modeling using finite difference method, which is named MiniMod, a mini application for seismic modeling. The purpose is to provide a benchmark suite that is, on one hand easy to build and adapt to the state of the art in programming models and changing high performance hardware landscape. On the other hand, the intention is to have a proxy application to actual production geophysical exploration workloads for Oil \& Gas exploration, and other geosciences applications based on the wave equation. From top to bottom, we describe the design concepts, algorithms, code structure of the application, and present the benchmark results on different current computer architectures.


## 1 Introduction

Minimod is a Finite Difference-based proxy application which implements seismic modeling (see Chapter 2) with different approximations of the wave equation (see Chapter 3). Minimod is selfcontained and designed to be portable across multiple High Performance Computing (HPC) platforms. The application suite provides both non-optimized and optimized versions of computational kernels for targeted platforms (see Chapter 5). The target specific kernels are provided in order to conduct benchmarking and comparisons for emerging new hardware and programming technologies.

Minimod is designed to:

- Be portable across multiple software stacks.
- Be self-contained.
- Provide non-optimized version of the computational kernels.
- Provide optimized version of computational kernels for targeted platforms.
- Evaluate node-level parallel performance.
- Evaluate distributed-level parallel performance.

${ }^{*}$ Total EP R\&T, email: jie.meng@total.com

${ }^{\dagger}$ Total S.A.

The first four items are covered in Section 5 and the remainder items are covered in Section 6 .

New HPC technologies evaluation is a constant task that plays a key role when decisions are taken in terms of computing capacity acquisitions. Evaluations in the form of benchmarking provide information to compare competing technologies wrt relevant workloads. Minimig is also use for this purpose, and insight collected with it has been part the last major acquisitions by Total (see Figure 1 .

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-02.jpg?height=952&width=1377&top_left_y=611&top_left_x=379)

Figure 1: Evolution of computing capacity wrt geophysical algorithms.

It can be observed in Figure 1, how more complex geophysical algorithms drive larger capacity installations in Total. The main driver of performance demand by the geophysical algorithms presented in figure are: the accuracy of the wave equation approximation and the addition of optimization or inverse problem schemes. The former is captured in Minimig, where the later is out of scope of this article. Performance trends obtained by conducting experiments with Minimig (or similar tools) influenced the decisions for the last ten years, this mainly motivated by the transition of the main workloads from Ray-based to wave-based methods.

## 2 Seismic Modeling

Seismic depth imaging is the main tool used to extract information describing the geological structures of the subsurface from recorded seismic data, it is effective till certain depth after which it becomes inaccurate. At its core it is an inverse problem which consists in finding the best model minimizing the distance between the observed data (recorded seismic data) and the predicted data (produced by computational means). The process to estimate the predicted data is known as forward modeling. It is based on the resolution of the wave equation for artificial perturbations of the subsurface given initial and boundary conditions. This simulation is repeated as many times as perturbations were introduced during seismic data acquisition. In Figure 2 on of such experiments is represented, in this case for a marine setup. The perturbation (namely source) is introduce by an airgun dragged behind a ship, then the waves propagate through the medium. At each interface between layers of materials with different characteristics part of the energy is reflected. These reflections are recorded at sea level (at surface for a onshore setup) by a network of sensors (in the figure depicted in red) also pulled by the ship.

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-03.jpg?height=531&width=957&top_left_y=976&top_left_x=584)

Figure 2: The mechanical medium represented by the subsurface model is perturbed and the wave propagation simulated. In this case, waves are depicted in discretized form as rays, for simplicity.

Solving the forward modeling efficiently is crucial for geophysical imaging as one needs to get solutions for many sources and many iterations as we progressively the subsurface model improves. Constant progresses in data acquisition and in rocks physics labs, more powerful computers and integrated team including physicists, applied mathematicians and computer scientists have greatly contributed to the development of advanced numerical algorithms integrating more and more complex physics. For the last 20 years, the field has been very active in the definition and introduction of different wave equation approximations and corresponding numerical methods for solving forward problem. But the real change came with the implementation of the full wave equation, thanks to the petascale era in HPC, giving access to a complete representation of the wavefield. It allowed geo-scientist to re-design imaging algorithm both in time dynamic and time harmonic domain. The most popular numerical scheme used, nowadays by the industry, is based on finite difference methods (FDM) on regular grids [9], 4]. We refer to [19] for examples of FDM in the geophysics frameworks and to [8] for 3D applications.

## 3 Finite Differences

Various numerical methods have been explored for the modeling of seismic wave propagation including the finite difference, finite element, finite volume and hybrid methods. Among those methods, the finite difference method is the most popular one for its simplicity and easy and straightforward implementation.

The first step of implementing the governing equations is to called discretizations, basically consist on write the equations on forms that allow the direct implementation of differential operators. The discretizations of the governing equations are impose on three different kind of grids, depending on the symmetry of the problem. We use the standard collocated grid, and two versions of staggered grid, namely Yee [21, 17], 18] and Lebedev [11].

The first equation to be described is the second order acoustic wave equation with constant density, solving for the pressure wavefield $p$,

$$
\begin{equation*}
\frac{1}{v_{p}^{2}} \frac{\partial^{2} p(\mathbf{x}, t)}{\partial t^{2}}-\nabla^{2} p(\mathbf{x}, t)=f(\mathbf{x}, t) \tag{1}
\end{equation*}
$$

where $v_{p}$ is the velocity of the pressure wavefield, $p(\mathbf{x}, t)$ expanded to $3 \mathrm{D}$ domain is $p(x, y, z, t)$, likewise for the source $f(\mathbf{x}, t)=f(x, y, z, t)$.

The second equation is the first order acoustic wave equation with variable density $\rho$,

$$
\begin{equation*}
\frac{1}{\rho v_{p}^{2}} \frac{\partial p(\mathbf{x}, t)}{\partial t}-\nabla \cdot \mathbf{v}(\mathbf{x}, t)=f(\mathbf{x}, t), \quad \rho \frac{\partial \mathbf{v}(\mathbf{x}, t)}{\partial t}-\nabla p(\mathbf{x}, t)=0 \tag{2}
\end{equation*}
$$

where $p$ is the pressure wavefield, and $\mathbf{v}$ is a vector wavefield for the particle velocities (time derivatives of displacement) along the different coordinate axis.

The third equation is the acoustic transversely isotropic first order system, see 3] for details.

Finally, we have the elastic equations with variable density $\rho$,

$$
\begin{equation*}
\frac{\partial \boldsymbol{\sigma}(\mathbf{x}, t)}{\partial t}-C D \mathbf{v}(\mathbf{x}, t)=\mathbf{f}(\mathbf{x}, t), \quad \rho \frac{\partial \mathbf{v}(\mathbf{x}, t)}{\partial t}-D^{t} \boldsymbol{\sigma}(\mathbf{x}, t)=0 \tag{3}
\end{equation*}
$$

where $\boldsymbol{\sigma}$ is a vector wavefield for the stresses using Voigt notation and $\mathbf{v}$ is a vector wavefield for the particle velocities. The derivative operator $D$ is

$$
D=\left(\begin{array}{ccc}
\frac{\partial}{\partial x} & &  \tag{4}\\
& \frac{\partial}{\partial y} & \\
& \frac{\partial}{\partial z} & \frac{\partial}{\partial z} \\
\frac{\partial}{\partial z} & & \frac{\partial}{\partial x} \\
\frac{\partial}{\partial y} & \frac{\partial}{\partial x} &
\end{array}\right)
$$

and $D^{t}$ is the transpose of $D$ without transpose of the derivatives. This is a subtle difference since a derivative is anti-symmetric. We have two different symmetry classes, isotropic and transversely
isotropic, which only differs in the sparsity pattern of the stiffness tensor $C$.

The above described discretizations are implemented with the following names as kernels:

- Acoustic_iso_cd: Standard second order acoustic wave-propagation in isotropic media with constant density.
- Acoustic_iso: first order acoustic wave-propagation in isotropic media on a staggered Yee-grid variable density.
- Acoustic_tti: first order acoustic wave-propagation in transversely isotropic media on a staggered Lebedev-grid.
- Elastic_iso: first order elastic wave-propagation in isotropic media on a staggered Yee-grid.
- Elastic_tti: first order elastic wave-propagation in transversely isotropic media on a staggered Lebedev-grid.
- Elastic_tti_approx: Non-standard first order elastic wave-propagation in transversely isotropic media on a staggered Yee-grid

All discretizations use CPML 10 at the boundary of the computational domain, with the option of using free surface boundary conditions at the surface. Full unroll of the discretization is given for acoustic_iso_cd, as example, this is the simplest kernel in Minimod for simulating acoustic wavepropagation in isotropic media with a constant density domain, i.e. equation (1). The equation is discretized in time using a second-order centered stencil, resulting in the semi-discretized equation:

$$
\begin{equation*}
p^{n+1}-Q p^{n}+p^{n-1}=\left(\Delta t^{2}\right) v_{p}^{2} f^{n} \tag{5}
\end{equation*}
$$

where

$$
Q=2+\Delta t^{2} v_{p}^{2} \nabla^{2}
$$

The equation is discretized in space using a 25 -point stencil in space, with nine points in each direction of three dimensions:

$$
\begin{array}{r}
\nabla^{2} p(x, y, z) \approx \sum_{m=1}^{4} c_{x m}[p(i+m, j, k)+p(i-m, j, k)-2 p(i, j, k)] \\
c_{y m}[p(i, j+m, k)+p(i, j-m, k)-2 p(i, j, k)] \\
c_{z m}[p(i, j, k+m)+p(i, j, k-m)-2 p(i, j, k)]
\end{array}++
$$

where $c_{x m}, c_{y m}$ and $c_{z m}$ are discretization parameters that approximates second derivatives in the different spatial directions. The parameters can be derived from the Taylor expansion of the derivatives in the $\mathrm{x}, \mathrm{y}$ and $\mathrm{z}$-direction respectively, where the approximation would be of order 8 . The derivatives can also use optimized stencils, that reduce the dispersion error at the expense of formal order.

## 4 Computing costs

Being the core algorithm of Finite Difference, stencil-based computation algorithms represent the kernels of many well-known scientific applications, such as geophysics and weather forecasting.

However, the peak performance of stencil-based algorithms are limited because of the imbalance between computing capacity of processors units and data transfer throughput of memory architectures. In Figure 3 the memory access problem is shown. The computing part of the problem is basically the low re-use of the memory accessed elements.

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-06.jpg?height=648&width=873&top_left_y=711&top_left_x=626)

Figure 3: Memory layout for an simple stencil example, to access the required values multiple useless cache lines (bottom) need to be accessed with incurred penalties. Figure extracted from [6].

In order to deal with the above described limitation, a great amount of research have been devoted to optimize stencil computations to achieve higher performance. For example, de la Cruz and Araya-Polo proposed the semi-stencil algorithm 6] which improves memory access pattern and efficiently reuses accessed data by dividing the computation into several updates. Rivera et al. [16] showed that tiling 3D stencil combined with array padding could significantly reduce miss rates and achieve performance improvements for key scientific kernels. Recently, Nguyen et al. [14] introduced higher dimension cache optimizations.

Advanced programming models have been explored to improve stencil performance and productivity. In 2012, Ghosh et al. 77 analyzed the performance and programmability of three high-level directive-based GPU programming models (PGI, CAPS, and OpenACC) on an NVIDIA GPU for kernels of the same type as described in previous sections and for Reverse Time Migration (RTM, [1), widely used method in geophysics. In 2017, Qawasmeh et al. [15] implemented an MPI plus OpenACC approach for seismic modeling and RTM. Domain-specific languages (DSLs) for stencil algorithms have also been proposed. For example, Louboutin et al. introduced Devito [12], which a new domain-specific language for implementing differential equation solvers. Also, de la Cruz and Araya-Polo proposed an accurate performance model for a wide range of stencil sizes which captures the behavior of such 3D stencil computation pattern using platform parameters [5].

## 5 Minimod Description

### 5.1 Source Code Structure

In this section, we introduce the basic structure of the source code in Minimod. As we described in Section 3, the simulation in Minimod consists of solving the wave equation, the temporal requires the spatial part of the equation to be solve at each timestep for some number of timesteps. The pseudo-code of the algorithm is shown in algorithm 1, for the second order isotropic constant density equation. We apply a Perfectly Matched Layer (PML) 2] boundary condition to the boundary regions. The resulting domain consists of an "inner" region where Equation 5 is applied, and the outer "boundary" region where a PML calculation is applied.

```
Data: f: source
Result: $\mathbf{p}^{n}$ : wavefield at timestep $n$, for $n \leftarrow 1$ to $T$
$1 \mathbf{p}^{0}:=0$;
for $n \leftarrow 1$ to $T$ do
    for each point in wavefield $\mathbf{u}^{n}$ do
        Solve Eq. 5 (left hand side) for wavefield $\mathbf{p}^{n}$;
    end
    $\mathbf{p}^{n}=\mathbf{p}^{n}+\mathbf{f}^{n}$ (Eq. 5 right hand side);
end
```

Algorithm 1: Minimod high-level description

As described in algorithm 1, the most computationally expensive component of minimod is the computation of the wavefield for each point. We list the code structure of the wavefield calculation in algorithm 2.

```
Data: $p^{n-1}, p^{n-2}$ : wavefields at previous two timsteps
Result: $p^{n}$ : wavefield at current timestep
for $i \leftarrow \mathrm{xmin}$ to xmax do
    if $i \geq \mathrm{x} 3$ and $i \leq \mathrm{x} 4$ then
        for $j \leftarrow$ ymin to ymax do
            if $j \geq \mathrm{y} 3$ and $j \leq \mathrm{y} 4$ then
                // Bottom Damping (i, j, z1...z2)
                // Inner Computation (i, j, z3...z4)
                // Top Damping (i, j, z5...z6)
            else
                // Back and Front Damping (i, j, zmin...zmax)
            end
        end
    else
        // Left and Right Damping (i, ymin...ymax, zmin...zmax)
    end
end
```

Algorithm 2: Wavefield solution step

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-08.jpg?height=593&width=523&top_left_y=240&top_left_x=736)

Figure 4: Code tree structure of Minimod package.

The general structure listed above is the backbone for all the propagators included in Minimod. To keep the code simple and flexible, each propagator is compiled separately. This can be selected by setting the propagator variable in the version file before compiling. Figure 4 presents a tree structure of Minimod code suite.

### 5.2 Targets

Each propagator has also its own implementation depending the hardware targeted. The target directory located in each propagator is the way of setting targets. In the source code structure, the following implementations of the target are provided:

- seq_opt-none : implement kernels without any optimization (so as the short form of sequential none optimization mode). The purpose of the sequential implementation is to be used as a baseline to compare the optimized results. Also, this is useful to analyze not parallel-related characteristics.
- omp_opt-none : implement kernels using OpenMP directives for multi-threading execution on CPU. The goal of this target implementation is to explore the advanced CPU features on multi-core or many-core systems.
- acc_opt-gpu : implement kernels using OpenACC directives for offloading computing to GPU (so as the short form of accelerator optimization using GPU). The goal of this target implementation is to explore the accelerating on GPU using OpenACC programming model standard.
- omp_offload_opt-gpu : implement kernels using OpenMP directives for offloading to GPU (so as the short form of OpenMP offloading optimization using GPU). The goal of implementing this target is to explore the accelerating on GPU using OpenMP programming standard.

In addition to change the propagator that is to be used in tests, one may also change the "version" file to use a different target by setting the "target" variable to the desired multi-threaded or accelerator implementations.

### 5.3 Compilation and usage

After the target propagators, compilers, and accelerating implementation settings are selected, the source code is ready for compilation, as follows:

```
To compile the sequential mode of Minimod package:
$> source set_env.sh
$> make
To compile the multi-threading mode with OpenMP directives:
$> source set_env.sh
$> make _USE_OMP=1
To compile the offloading to GPU mode with OpenMP directives:
$> source set_env.sh
$> make _USE_OMP=tesla
To compile the multi-threading mode with OpenACC directives:
$> source set_env.sh
$> make _USE_ACC=multicore
To compile the offloading to GPU mode with OpenACC directives:
$> source set_env.sh
$> make _USE_ACC=tesla
```

The parameters of Minimod are shown in the following verbatim section. Those are the basic parameters for seismic modeling and they are set as command-line options. The main parameters include: grid sizes, grid spacing on each dimension, the number of time steps and the maximum frequency.

```
[]$ ./modeling_acoustic_iso_cd_seq_opt-none --help
```

| --ngrid | $100,100,100$ | \# Grid size |
| :--- | :--- | :--- |
| --dgrid | $20,20,20$ | \# Dmesh: grid spacing |
| --nsteps | 1000 | \# Nb of time steps for modeling |
| --fmax | 25 | \# Max Frequency |
| --verbose | .false. | \# Print informations |

In terms of expected results, the following verbatim section presents an example to show how to run the application and the run-time results of single-thread Minimod acoustic-iso-cd kernel. As we can see, the results report all the parameters that are used in the modeling and at the end the kernel time and modeling time of running the application.

[]]\$ ./modeling_acoustic_iso_cd_seq_opt-none --ngrid 240,240,240 --nsteps 300

| nthreads | $=$ | 1 |  |  |
| :---: | :---: | :---: | :---: | :---: |
| ngrid | $=$ | 240 | 240 | 240 |
| dgrid | $=$ | 20.0000000 | 20.0000000 | 20.0000000 |
| nsteps | $=$ | 300 |  |  |
| fmax | $=$ | 25.0000000 |  |  |
| vmin | $=$ | 1500.00000 |  |  |
| vmax | $=$ | 4500.00000 |  |  |
| $\mathrm{cfl}$ | $=$ | 0.800000012 |  |  |
| stencil | $=$ | 4 | 4 | 4 |
| source_loc | $=$ | 120 | 120 | 120 |
| ndamping | $=$ | 27 | 27 | 27 |
| ntaper | $=$ | 3 | 3 | 3 |
| nshots | $=$ | 1 |  |  |
| time_rec | $=$ | 0.00000000 |  |  |
| nreceivers | $=$ | 57600 |  |  |
| receiver_increment | $t=$ | 1 | 1 |  |
| source_increment | $=$ | 1 | 1 | 0 |
| time step | $100 /$ | 300 |  |  |
| time step | 200 / | 300 |  |  |
| time step | $300 /$ | 300 |  |  |
| Time Kernel | 30.47 |  |  |  |
| Time Modeling | 31.01 |  |  |  |

## 6 Benchmarks

In this section examples of Minimod experimental results are presented. The purpose is illustrate performance and scalability of the propagators with regard to current HPC platforms.

### 6.1 Experimental set-up

The different propagators of Minimod are evaluated on Fujitsu A64FX architecture, AMD EYPC system, Intel Skylake and IBM Power8 system, as well as Nvidia's V100 GPUs. The specifications of hardware and software configurations of the experimental platforms are shown in Table 1 .

|  | Hardware | Software |
| :---: | :---: | :---: |
| CPUs | A64FX Armv8-A SVE architecture | Fujitsu Compiler 1.1.13 (frt) |
| CPU cores | 48 computing cores | OpenMP (-Kopenmp) |
| Memory | 32 GB HBM2 | auto-parallelisation |
| L2 | $8 \mathrm{MB}$ | (-Kparallel) |
| L1 | $64 \mathrm{~KB}$ |  |
| Device Fabrication | $7 \mathrm{~nm}$ |  |
| TDP | $160 \mathrm{~W}$ |  |
| CPUs | AMD EYPC 7702 | GCC 8.2 (gfortran) |
| CPU cores | 64 computing cores | OpenMP |
| Memory | $256 \mathrm{~GB}$ |  |
| $\mathrm{L} 3$ | $256 \mathrm{MB}$ (per socket) |  |
| L2 | $32 \mathrm{MB}$ |  |
| L1 | $2+2 \mathrm{MB}$ |  |
| Device Fabrication | $14 \mathrm{~nm}$ |  |
| TDP | $200 \mathrm{~W}$ |  |
| CPUs | 2x Intel Xeon Gold 5118 | intel compiler 17.0.2 (ifort) |
| CPU cores | 24 (12 per $\mathrm{CPU})$ |  |
| Memory | $768 \mathrm{~GB}$ |  |
| L3 | $16 \mathrm{MB}$ (per socket) |  |
| L2 | $1024 \mathrm{~KB}$ |  |
| L1 | $32+32 \mathrm{~KB}$ |  |
| Device Fabrication | $14 \mathrm{~nm}$ |  |
| $\mathrm{TDP}$ | $2 \times 105 \mathrm{~W}$ |  |
| CPUs | 2 x IBM Power8 (ppc64le) | PGI 19.7 (pgfortran) |
| CPU cores | 20 computing cores (10 per CPU) | OpenMP (-mp) |
| Memory | $256 \mathrm{~GB}$ |  |
| L3 | $8 \mathrm{MB}$ (per two cores) |  |
| L2 | $512 \mathrm{~KB}$ (per two cores) |  |
| L1 | $64+32 \mathrm{~KB}$ |  |
| Device Fabrication | $22 \mathrm{~nm}$ |  |
| TDP | $2 \times 190 \mathrm{~W}$ |  |
| GPU | 1 x Nvidia V100 | PGI 19.7 (pgfortran) |
| cores | 2560 Nvidia CUDA cores | OpenACC (-ta=tesla) |
| Memory | 16 GB HBM2 |  |
|  | $6144 \mathrm{~KB}$ |  |
| Device fabrication | $12 \mathrm{~nm}$ FFN |  |
| Power consumption | $290 \mathrm{~W}$ |  |

Table 1: Hardware and software configuration of the experimental platforms. From top to bottom, the first section is Fujitsu A64FX Arm8-A architecture. The second section is AMD EYPC Rome architecture. The third section is Intel Skylake architecture. The fourth section is IBM Power8 architecture. And the bottom section is the specification of Nvidia's V100 GPU.

### 6.2 Performance characterization

In our experiments, we use roofline model proposed by Williams et al. [20] to understand the hardware limitations as well as evaluating kernel optimization. In the roofline model, the performance of various numerical methods are upper bounded by the peak floating point operations (flop) rate and the memory bandwidth while running on single-core, multi-core or accelerator processor architecture.

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-12.jpg?height=794&width=1455&top_left_y=584&top_left_x=324)

Figure 5: Roofline model analyses on AMD EYPC System.

Figure 5 shows the peak performance in term of GFlops per seconds and memory bandwidth of cache and main dynamic random-access memory (DRAM) of the AMD EYPC system listed in Table 1 where we conducted experiments on. The arithmetic intensity in the roofline plot is calculated by the number of floating point operations that are performed in the stencil calculation divided by the number of words that we need to read from and write to memory [6].

### 6.3 Single compute node-level parallelism

We use Minimod to experiment the single compute node-level parallelism on different computer systems. As shown in Figure 7. The system-level performance tests are conducted on IBM power, Fujitsu A64FX systems, and compared with using NVIDIA's V100 GPUs as accelerators. The different propagators in Minimod (acoustic_iso_cd, acoustic_iso, acoustic_tti, elastic_iso, and elastic_tti) are tested, and results are shown in Figure 6 .

As we observe in Figure 6, the Fujitsu A64FX processor (as shown in the orange bars) provides better performance for all the propagators in comparison to both IBM power system (as shown in the dark blue bars), Intel skylake system (as shown in the light blue bars), as well as AMD EYPC Rome systems (as shown in the yellow bars). In fact, the performance of Fujitsu A64FX is closer
to the performance of the system with Nvidia's V100 GPU accelerator (as shown in the green bars).

The single node-level scalability tests are conducted on IBM power, AMD EYPC, and Fujitsu A64FX systems. The problem size for the strong scalability tests are set to be $240 \times 240 \times 240$. As presented in Figure 7, the results are compared between the three modern computer systems and also compares against the ideal case. Across the three systems, Fujitsu A64FX system again wins IBM power and AMD EYPC Rome systems in the single-node scalability tests.

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-13.jpg?height=780&width=1521&top_left_y=610&top_left_x=302)

Figure 6: System-level performance comparison results of Minimod with different propagators running on IBM power system (dark blue bars), on Intel Skylake system (light blue bars), on AMD EPYC system (yellow bars), on Fujitsu A64FX system (orange bars), and on NVIDIA's V100 GPU (green bars).

### 6.4 Distributed Memory Approach

The distributed version of Minimod is implemented using Message Passing Interface (MPI). The domain decomposition is defined using regular Cartesian topology, and the domain decomposition parameters need to match the total number of MPI ranks: for example, for the three-dimensional domain decomposition in $x \times y \times z$ equals $2 \times 2 \times 4$, the rank number needs to be 16 . As for the time being, only acoustic_iso_cd propagator is available within the distributed version of Minimod.

The implementation of MPI communication between sub-domains uses non-blocking send and receives. The communication operates in "expected message" mode that has no overlap of communication with computation. Each subdomain performs the following steps: first, to pack the messages to be transmitted in buffers; second, to perform communication by posting all sends and receives, and finally wait till the communication is complete and unpacks the arrived data.

We evaluated both weak scalability and strong scalability of the distributed version of Minimod

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-14.jpg?height=683&width=1198&top_left_y=244&top_left_x=472)

Figure 7: Single compute node-level scalability comparison results of Minimod running on IBM power system (blue curve), on AMD EYPC Rome system (yellow curve), and on Fujitsu A64FX system (red curve), and both are compared against the ideal scale-up (green curve).
![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-14.jpg?height=578&width=1654&top_left_y=1126&top_left_x=234)

Figure 8: MPI weak scalability of Minimod running on IBM power system for both the ideal problem-sizes (baseline $1000 \times 1000 \times 1000$ with linear increment on $\mathrm{X}$ dimension) and practical problem-sizes (the problem-size increments are balanced on each dimension) running on 1 to $8 \mathrm{MPI}$ ranks respectively.

for acoustic_iso_cd propagator. The results of weak scalability running Minimod on IBM power system is shown in Figure 8, which presents the evaluation results using both the ideal problem sizes and the practical problem sizes running on 1 to 8 MPI ranks, respectively.

For the weak scalability test running ideal problem sizes, we used a baseline of $1000 \times 1000 \times 1000$ with linear increment on $\mathrm{X}$ dimension (for example, for the test running on 6 MPI ranks we used a problem-size of $6000 \times 1000 \times 1000$ ). And for practical problem-sizes, we used the same baseline while the problem-size increments are balanced on each dimension (for example, for the test run-
ning on 6 MPI ranks we used a problem-size of $1856 \times 1856 \times 1856$ ). The green curves in Figure 8 present the efficiencies in comparison to the baseline result.

We can see from Figure 8 that the weak scalability holds well for running from 1 rank scale to up to 8 ranks for the ideal problem sizes. And for the practical problem sizes which is more close to the real seismic acquisition situation, the weak scalability efficiencies for 2 ranks and 4 ranks are higher than $100 \%$ because of the slightly smaller problem sizes compared to the baseline case $(1280 \times 1280 \times 1280$ for 2 ranks and $1600 \times 1600 \times 1600$ for 4 ranks $)$, while it starts diminishing when it reaches 8 ranks mainly because of the increase of problem sizes.

The results of strong scalability are shown in Figure 9. The strong scalability tests are conducted on both IBM power and Fujitsu A64FX systems. The problem size for the strong scalability tests is set to $1024 \times 1024 \times 1024$, on the rank numbers of $8,16,32,64,128$, and 256 respectively.

As presented in Figure 9, the results of the kernel execution on the IBM power and the Fujitsu A64FX systems are compared with the ideal scaling trend. The strong scalability results on both systems are very close when the MPI rank number is smaller than 64 , while the kernel shows slightly better scalability results on the IBM system than on the Fujitsu system when running with 128 and 256 MPI ranks. In comparison to the ideal case, scalability on the IBM power system reached $63 \%$ while on the Fujitsu system reached $60 \%$ of the ideal scalability.

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-15.jpg?height=813&width=1353&top_left_y=1290&top_left_x=386)

Figure 9: MPI strong scalability comparison results of Minimod running on IBM power system (blue curve) and on Fujitsu A64FX system (red curve), and both are compared against the ideal scale-up (green curve).

### 6.5 Profiling

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-16.jpg?height=1130&width=1651&top_left_y=351&top_left_x=237)

Figure 10: Profiling results of Minimod using HPCToolkit.

Profiling and analyses was conducted on Minimod, for example using the HPCToolkit [13] from Rice University. Figure 10 shows a screenshot of the trace view in HPCToolkit profiling Minimod acoustic iso kernel implemented in multi-threading mode using OpenMP. The biggest panel on the top left presents sequences of samples of each trace line rendered. The different colors represent the time spends on different subroutines which are listed on the right panel. The bottom panel in Figure 10 is the depth view of the target Minimod application which presents the call path at every time step.

As an illustrative example for profiling Minimod, Figure 11 shows the profiling results from HPCToolkit trace view for the sequential implementation of the simplest kernel acoustic_iso_cd (acoustic wave-propagation in isotropic media with constant density) in Minimod without any optimization. To better understand the behavior of the kernel, what is shown in the picture is a case with one thread with the stencil computation on a grid size of $100 \times 100 \times 100$. As shown in Figure 11, the majority of the time is spent on running the "target_pml_3d" which is the implementation of perfectly-matched region, as shown in the dark color areas in the top left panel. And the green vertical line is for the "target_inner_3d", where the thread performs computation for the inner region of stencil.

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-17.jpg?height=1098&width=1635&top_left_y=356&top_left_x=234)

Figure 11: Example of profiling sequential mode of Minimod acoustic-iso-cd kernel using HPCToolkit.

An advantage of HPCToolkit it that can profile the results of Minimod GPU mode for each time sampling traces. Figure 12 shows the the profiling results of the entire execution of Minimod acoustic-iso-cd kernel in OpenACC offloading to GPU mode. Different than the CPU profiling trace views, the GPU profiling trace view on HPCToolkit top-left panel window is composed of two rows. The top row shows the CPU (host) thread traces and the bottom row is for the GPU (device) traces.

A zoomed-in view of this GPU profiling results is presented in Figure 13. We selected time step shows the GPU that is running the "target_pml_3d" kernel where the blank white spaces in the GPU row shows the idleness. The same as in the profiling results for CPU, different colors here represent the time spends on different GPU calls.

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-18.jpg?height=835&width=1569&top_left_y=274&top_left_x=278)

Figure 12: GPU profiling results of Minimod acoustic-iso-cd kernel using HPCToolkit.

![](https://cdn.mathpix.com/cropped/2024_06_04_a2a689c4c555ef3e09feg-18.jpg?height=840&width=1567&top_left_y=1271&top_left_x=279)

Figure 13: A detailed view of GPU profiling Minimod acoustic-iso-cd kernel using HPCToolkit.

## 7 Conclusion

This article introduces a proxy application suite for seismic modeling using finite difference method named Minimod. The design concepts, underline algorithms, and code structures of Minimod are described. The benchmark results of Minimod are shown on different computer architectures for both single compute node-level parallelism and distributed memory approaches.

## 8 Acknowledgements

We would like to thank Total and subsidiaries for allowing us to share this material. We would also like to express our appreciation to Diego Klahr for his continuous support, and our colleague Elies Bergounioux in France for discussions on the adaptability of proxy applications in production. We also thank Ryuichi Sai from Rice University for his contribution on the profiling results using HPCToolkits. We would like acknowledge Pierre Lagier from Fujitsu for his help with the experiments conducted with latest Fujitsu technology. Last but not least, many thanks to our former colleague Maxime Hugues for his initial implementation of the presented software.

## References

[1] M. Araya-Polo, J. Cabezas, M. Hanzich, M. Pericas, F. Rubio, I. Gelado, M. Shafiq, E. Morancho, N. Navarro, E. Ayguade, J. M. Cela, and M. Valero. Assessing accelerator-based hpc reverse time migration. IEEE Transactions on Parallel and Distributed Systems, 22(1):147-162, 2011

[2] J.-P. Berenger. A perfectly matched layer for the absorption of electromagnetic waves. Journal of Computational Physics, 114(2):185 - 200, 1994.

[3] K. Bube, T. Nemeth, P. Stefani, R. Ergas, W. Lui, T. Nihei, and L. Zhang. On the instability in second-order systems for acoustic vti and tti media. Geophysics, 77:171-186, 2012.

[4] M. Dablain. The application of high-order differencing to the scalar wave equation. Geophysics, $51: 54-66,1986$.

[5] R. de la Cruz and M. Araya-Polo. Towards a multi-level cache performance model for 3d stencil computation. Procedia Computer Science, 4:2146 - 2155, 2011. Proceedings of the International Conference on Computational Science, ICCS 2011.

[6] R. de la Cruz and M. Araya-Polo. Algorithm 942: Semi-stencil. ACM Trans. Math. Softw., 40(3), Apr. 2014.

[7] S. Ghosh, T. Liao, H. Calandra, and B. M. Chapman. Experiences with openmp, pgi, hmpp and openacc directives on iso/tti kernels. In 2012 SC Companion: High Performance Computing, Networking Storage and Analysis, pages 691-700, Nov 2012.

[8] R. W. Graves. Simulating seismic wave propagation in 3d elastic media using staggered-grid finite differences. Geophysics, 86:1091-1106, 1996.

[9] K. R. Kelly, R. W. Ward, S. Treitel, and R. M. Alford. Synthetic seismograms: A finitedifference approach. Geophysics, 41:2-27, 1976.

[10] D. Komatitsch and R. Martin. An unsplit convolutional perfectly matched layer improved at grazing incidence for the seismic wave equation. Geophysics, 72:155-167, 2007.

[11] V. Lebedev. Difference analogues of orthogonal decompositions, basic differential operators and some boundary problems of mathematical physics. ii. USSR Computational Mathematics and Mathematical Physics, 4:36-50, 1964.

[12] M. Louboutin, M. Lange, F. Luporini, N. Kukreja, P. A. Witte, F. J. Herrmann, P. Velesko, and G. J. Gorman. Devito (v3.1.0): an embedded domain-specific language for finite differences and geophysical exploration. Geoscientific Model Development, 12(3):1165-1187, 2019.

[13] J. Mellor-Crummey, R. Fowler, and D. Whalley. Tools for application-oriented performance tuning. In Proceedings of the 15th International Conference on Supercomputing, ICS '01, page 154-165, New York, NY, USA, 2001. Association for Computing Machinery.

[14] A. Nguyen, N. Satish, J. Chhugani, C. Kim, and P. Dubey. 3.5-d blocking optimization for stencil computations on modern cpus and gpus. In SC '10: Proceedings of the 2010 ACM/IEEE International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1-13, 2010.

[15] A. Qawasmeh, M. R. Hugues, H. Calandra, and B. M. Chapman. Performance portability in reverse time migration and seismic modelling via openacc. The International Journal of High Performance Computing Applications, 31(5):422-440, 2017.

[16] G. Rivera and Chau-Wen Tseng. Tiling optimizations for 3d scientific computations. In $S C$ '00: Proceedings of the 2000 ACM/IEEE Conference on Supercomputing, pages 32-32, 2000.

[17] J. Virieux. Sh-wave propagation in heterogeneous media: Velocity-stress finite-difference method. Geophysics, 49:1933-1957, 1984.

[18] J. Virieux. P-sv wave propagation in 'heierogeneous media: velocity-stress finite-difference method. Geophysics, 51:889-901, 1986.

[19] J. Virieux and S. Operto. An overview of full-waveform inversion in exploration geophysics. Geophysics, 74, 2009.

[20] S. Williams, A. Waterman, and D. Patterson. Roofline: An insightful visual performance model for multicore architectures. Commun. ACM, 52:65-76, 042009.

[21] K. Yee. "numerical solution of initial boundary value problems involving maxwell's equations in isotropic media.". IEEE Transactions on Antennas and Propagation, 14:302-307, 1966.

</end of paper 0>


<paper 1>
# Massively scalable stencil algorithm 

Mathias Jacquelin 8<br>Cerebras Systems Inc.<br>Sunnyvale, California, USA<br>mathias.jacquelin@cerebras.net

Mauricio Araya-Polo 8 and Jie Meng<br>TotalEnergies EP Research \& Technology US, LLC.<br>Houston, Texas, USA<br>mauricio.araya@totalenergies.com


#### Abstract

Stencil computations lie at the heart of many scientific and industrial applications. Unfortunately, stencil algorithms perform poorly on machines with cache based memory hierarchy, due to low reuse of memory accesses. This work shows that for stencil computation a novel algorithm that leverages a localized communication strategy effectively exploits the Cerebras WSE-2, which has no cache hierarchy. This study focuses on a 25 -point stencil finite-difference method for the 3D wave equation, a kernel frequently used in earth modeling as numerical simulation. In essence, the algorithm trades memory accesses for data communication and takes advantage of the fast communication fabric provided by the architecture. The algorithm - historically memory bound - becomes compute bound. This allows the implementation to achieve near perfect weak scaling, reaching up to 503 TFLOPs on WSE-2, a figure that only full clusters can eventually yield.

Index Terms-Stencil computation, high performance computing, energy, wafer-scale, distributed memory, multi-processor architecture and microarchitecture


## I. INTRODUCTION

Stencil computations are central to many scientific problems and industrial applications, from weather forecast ( [32]) to earthquake modeling ( [19]). The memory access pattern of this kind of algorithm, in which all values in memory are accessed but used in only very few arithmetic operations, is particularly unfriendly to hierarchical memory systems of traditional architectures. Optimizing these memory operations is the main focus of performance improvement research on the topic.

Subsurface characterization is another area where stencils are widely used. The objective is to identify major structures in the subsurface that can either hold hydrocarbon or be used for $\mathrm{CO}_{2}$ sequestration. One step towards that end is called seismic modeling, where artificial perturbations of the subsurface are modeled solving the wave equation for given initial and boundary conditions. Solving seismic modeling efficiently is crucial for subsurface characterization, since many perturbation sources need to be modeled as the subsurface model iteratively improves. The numerical simulations required by seismic algorithms for field data are extremely demanding, falling naturally in the HPC category and requiring practical evaluation[^0]

| Traditional architecture | WSE |
| :---: | :---: |
| L1 | Memory |
| L2 \& L3 | $\varnothing$ |
| DRAM | $\varnothing$ |
| Off-node interconnect | Fabric \& routers |

TABLE I: Equivalences between traditional architectures and the WSE

of technologies and advanced hardware architectures to speed up computations.

Advances in hardware architectures have motivated algorithmic changes and optimizations to stencil applications for at least 20 years ( [23]). Unfortunately, the hierarchical memory systems of most current architectures is not well-suited to stencil applications, therefore limiting performance. This applies to multi-core machines, clusters of multi-cores, and accelerator-based platforms such as GPGPUs, FPGAs, etc. ( [2], [5]). Alternatively, nonhierarchical architectures were explored in this context, such as the IBM Cell BE ( [3]), yielding high computational efficiency but with limited impact.

A key element for large scale simulations is the potential of deploying substantial number of processing units connected by an efficient fabric. The Cell BE lacked the former and it had limited connectivity. Another example of nonhierarchical memory system is the Connection Machine ( [12]), which excelled on scaling but at the cost of a very complex connectivity. In this work, a novel stencil algorithm based on localized communications that does not depend on memory hierarchy optimizations is introduced. This algorithm can take advantage of architectures such as the WSE from Cerebras ( [4]) and potentially Anton 3-like systems ( [28]). These are examples of architectures addressing both limitations described above.

Another angle to be considered is the availability of hardware-based solutions in the market. Literature review yields no generally available hardware architecture addressing the specific bottlenecks of stencil applications. Only a few custom designs examples are available ( [10], [14]).

In this work, an implementation of such seismic modeling method on a novel architecture is presented. The proposed mapping requires a complete redesign of the basic stencil algorithm. The contribution of this work is multi-fold:

- An efficient distributed implementation of Finite Differences for seismic modeling on a fully-distributed memory architecture with 850,000 processing elements.
- A stencil algorithm that is performance bound to the capacity of individual processing element rather than bound by memory or communication bandwidth.
- The target application ported relies on an industryvalidated stencil order.

The paper is organized as follows: Section II reviews relevant contributions in the literature. Section III describes the target application. Section IV provides details of how the target application was redesigned to efficiently use the novel processor architecture. Sections V and VI discusses experimental results and profiling data. Section VII provides discussions and conclusions.

## II. RELATED WORK

## A. Stencil Computation

Not all stencil computations are the same, and the structure and order of the stencil set the limits of the attainable performance. The higher the order (neighbors to be accounted for) and the closer to a pure star shape is, the harder to compute the stencil is. Traditional hierarchical memory subsystems will be overwhelmed by the memory access pattern which displays very little data reuse. Considerable amount of research effort has been devoted to optimizing stencil computations, and to finding ways around these issues. Spurred by emerging hardware technologies, studies on how stencil algorithms can be tailored to fully exploit unique hardware characteristics have covered many aspects, from DSLs, performance modeling, to pure algorithmic optimizations, targeting lowlevel architectural features in some cases.

Domain-specific languages (DSLs), domain-specific parallel programming models, and compiler optimizations for stencils have been proposed (e.g., [9], [11], [16], [22]). Performance models have been developed for this computing pattern (see [6], [30]), and the kernel has been ported to a variety of platforms ( [2], [3], [5], [35]) including specific techniques to benefit from unique hardware features.

Stencil computations have also been the subject of multiple algorithmic optimizations. Spatial and temporal blocking has been proposed [8], [13], [34]. A further example is the semi-stencil algorithm proposed by De la Cruz et al. [7], which offers an improved memory access pattern and a higher level of data reuse. Promising results are also achieved using a higher dimension cache optimization, as introduced by Nguyen et al. [20] , accommodating both thread-level and data-level parallelism. Most recently, Sai et al. [26] studied high-order stencils with a manually crafted collection of implementations of a 25 -point seismic modeling stencil in CUDA and HIP for the latest GPGPU hardware. Along this line of hardware-oriented stencil optimizations, Matsumura et al. [17] proposed a frame- work (AN5D) for GPU stencil optimization, obtaining remarkable results.

Wafer-scale computations have first been explored in Rocki et al. [24], in which the authors explore a BiCGStab implementation to solve a linear system arising from a 7-point finite difference stencil on the first generation of Cerebras Wafer-Scale Engine. Albeit computing a much simpler stencil and having a higher arithmetic intensity, this study paved the way to the work presented in this study. A notable difference between the current work and this study is that floating point operations were performed in mixed precision: stencil and AXPY operations being computed using 16 bit floating point operations and global reductions using 32 bit arithmetic. In the present study, only 32 bit floating point arithmetic is used, and neither AXPY operation nor global reductions are involved. This makes the performance achieved by these two applications not directly comparable.

## III. FinITE DIFFERENCE FOR SEISMIC MODELING

Minimod is a proxy application that simulates the propagation of waves through the Earth models, by solving a Finite Difference (FD) which is discretized form of the wave equation. It is designed and developed by TotalEnergies EP Research \& Technologies [18]. Minimod is self-contained and designed to be portable across multiple compilers. The application suite provides both nonoptimized and optimized versions of computational kernels for targeted platforms. The main purpose is benchmarking of emerging new hardware and programming technologies. Non-optimized versions are provided to allow analysis of pure compiler-based optimizations.

In this work, one of the kernels contained in Minimod is used as target for redesign: the acoustic isotropic kernel in a constant-density domain [21]. For this kernel, the wave equation PDE has the following form:

$$
\begin{equation*}
\frac{1}{\mathbf{V}^{2}} \frac{\partial^{2} \mathbf{u}}{\partial t^{2}}-\nabla^{2} \mathbf{u}=\mathbf{f} \tag{1}
\end{equation*}
$$

where $\mathbf{u}=\mathbf{u}(x, y, z)$ is the wavefield, $\mathbf{V}$ is the Earth model (with velocity as the main property), and $\mathbf{f}$ is the source perturbation. The equation is discretized in time using a $2^{\text {nd }}$ order centered stencil, resulting in the semidiscritized equation:

$$
\begin{gather*}
\mathbf{u}^{n+1}-\mathbf{Q} \mathbf{u}^{n}+\mathbf{u}^{n-1}=\left(\Delta t^{2}\right) \mathbf{V}^{2} \mathbf{f}^{n}  \tag{2}\\
\text { with } \mathbf{Q}=2+\Delta t^{2} \mathbf{V}^{2} \nabla^{2}
\end{gather*}
$$

Finally, the equation is discretized in space using a 25 point stencil in $3 \mathrm{D}$ ( $8^{\text {th }}$ order in space), with four points in each direction as well as the centre point:

$$
\begin{array}{r}
\nabla^{2} \mathbf{u}(x, y, z) \approx \sum_{m=0}^{4} c_{x m}[\mathbf{u}(i+m, j, k)+\mathbf{u}(i-m, j, k)]+ \\
c_{y m}[\mathbf{u}(i, j+m, k)+\mathbf{u}(i, j-m, k)] \\
c_{z m}[\mathbf{u}(i, j, k+m)+\mathbf{u}(i, j, k-m)]
\end{array}
$$

where $c_{x m}, c_{y m}, c_{z m}$ are discretization parameters, solved in step 4 in Algorithm 1. In the remainder of the document we refer to this operator as the Laplacian.

A simulation in Minimod consists of solving the wave equation at each timestep for thousands of timesteps. Pseudocode of the algorithm is shown in Algorithm 1.

```
Data: f: source
Result: $\mathbf{u}^{n}$ : wavefield at timestep $n$, for $n \leftarrow 1$ to
            $T$
$\mathbf{u}^{0}:=0$
for $n \leftarrow 1$ to $T$ do
    for each point in wavefield $\mathbf{u}^{n}$ do
Solve Eq. 2 (left hand side) for wavefield $\mathbf{u}^{n}$;
    end
    $\mathbf{u}^{n}=\mathbf{u}^{n}+\mathbf{f}^{n}$ (Eq. 2 right hand side);
end
```

Algorithm 1: Minimod high-level description

We note that a full simulation includes additional kernels, such as I/O and boundary conditions. These additional kernels are not evaluated in this study but will be added in the future. The kernel has been ported and optimized for GPGPUs, including NVIDIA A100, full report can be found in [25], this implementation is used as baseline to compare the results with the proposed implementation.

## IV. Finite-DifferenCES ON THE WSE

This section introduces general architectural details of the Cerebras Wafer-Scale Engine (WSE) and discusses hardware features allowing the target application to reach the highest level of performance. The mapping of the target algorithm onto the system is then discussed. The implementation is referred to as Finite Differences in the remainder of the study. Communication strategy and core computational parts involved in Finite Differences are also reviewed.

The implementation of Finite Differences on the WSE is written in Cerebras Software Language (CSL) using the Cerebras SDK [27], which allows software developers to write custom programs for Cerebras systems. CSL is a C-like language based on Zig [31], a reinterpretation of C which provides a simpler syntax and allows to declare compile-time blocks/optimizations explicitly (rather than relying on macros and the $\mathrm{C}$ preprocessor). CSL provides direct access to key hardware features, while allowing the use of higher-level constructs such as functions and while loops. The language allows to express computations and communications across multiple cores. Excerpts provided in the following will use the CSL syntax.

## A. The WSE architecture

The WSE is an unprecedented-scale manycore processor. It is the first wafer-scale system [4], embedding all compute and memory resources within a single silicon wafer, together with a high performance communication interconnect. An overview of the architecture is given in Figure 2. In its latest version, the WSE-2 provides a total of 850,000 processing elements, each with $48 \mathrm{~KB}$ of dedicated SRAM memory; up to eight 16-bit floating point operations per cycle; 16 bytes of read and 8 bytes of write bandwidth to the memory per cycle; and a 2D mesh interconnection fabric that can handle 4 bytes of bandwidth per PE per cycle in steady state [15].

The WSE can be seen as an on-wafer distributedmemory machine with a 2D-mesh interconnection fabric. This on-wafer network connects processing elements or PEs. Each PE has a very fast local memory and is connected to a router. The routers link to the routers of the four neighboring PEs. There is no shared memory. The WSE contains a $7 \times 12$ array of identical "dies", each holding thousands of PEs. Other chips are made by cutting the wafer into individual die. In the WSE, however, the interconnect is extended between dies. This results in a wafer-scale processor tens of times larger than the largest processors on the market at the time of its release.

The instruction set of the WSE is designed to operate on vectors or higher dimensionality objects. This is done by using data structure descriptors, which contain information regarding how a particular object should be accessed and operated on (such as address, length, stride, etc.).

As mentioned above, given the distributed memory nature of the WSE, the interconnect plays a crucial role in delivering performance. It is convenient to think of the 2D mesh interconnect in terms of cardinal directions. Each PE has 5 full-duplex links managed by its local router: East, West, North, and South links allow data to reach other routers and PEs, while the ramp link allows data to flow between the router and the $\mathrm{PE}$, on which computations can take place. Each link is able to move a 32 bit packet in each direction per cycle. Each unidirectional link operates in an independent fashion, allowing concurrent flow of data in multiple directions.

Every 32 bit packet has a color (contained in additional bits of metadata). The role of colors is twofold:

1) Colors are used in the routing of communications. A color determines the packet's progress at each router it encounters from source to destination(s). A router controls, for each color, where - to what subset of the five links to send a packet of that color. Moreover, colors are akin to virtual channels in which ordering is guaranteed.
2) Colors can also be used to indicate the type of a message: a color can be associated to a handler triggered when a packet of that particular color arrives.

The WSE is an unconventional parallel computing machine in the sense that the entire distributed memory machine lies within the same wafer. There is no cache hierarchy nor shared memory. Equivalences between hardware features of the WSE and what they correspond to

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-04.jpg?height=330&width=334&top_left_y=171&top_left_x=424)

Fig. 1: The 25-point stencil used in Finite Differences. Cells in white (along $Z$ ) reside in the local memory of a $\mathrm{PE}$, blue cells are communicated along the $X$ dimension, and green cells are communicated along the $Y$ dimension.

on traditional architectures (such as distributed memory supercomputers) are summarized in Table I.

## B. Target Algorithm/application mapping

The sheer scale of the WSE calls for a novel mapping of the target algorithm onto the processor. The 3D $n x \times n y \times n z$ grid on which the stencil computation is performed is mapped onto the WSE in two different ways: the $X$ and $Y$ dimensions are mapped onto the fabric while the $Z$ dimension is mapped into the local memory of a PE. This follows the approach that was explored in Rocki et al. [24], and has the benefit of expressing the highest possible level of concurrency for this particular application. Figure 3a depicts how the domain is distributed over the WSE. Each PE owns a subset of $n z$ cells of the original grid, as depicted in Figure 3b. In order to simplify the implementation, this local subset is extended by 8 extra cells: 4 cells below and 4 cells above the actual grid. This ensures that any cell in the original grid always has 4 neighbors below and 4 neighbors above. A PE stores the wavefield at two time steps (see Equation 2). In order to lower overheads, computations and communications are performed on blocks of size $b$. The block size is chosen to be the largest such that the $2 \times(n z+8)$ cells and all buffers depending on $b$ can fit in memory.

## C. Stencil computation

Computing the spatial component (referred to as the Laplacian) of the governing PDE lies at the heart of the target application, and it is traditionally the most demanding component. In addition to requiring a significant amount of floating point operations, computing the Laplacian also involves data movement, which is known to be very expensive on distributed memory platforms.

In the context of this paper, a 25-point stencil (depicted in Figure 1) is used. The stencil spans over all three dimensions of the grid. In order to compute a particular cell, data from neighboring cells is needed in all three dimensions. More precisely, a cell $l_{x, y, z}$ requires data from neighboring cells:

$$
\begin{array}{cl}
\text { cell }_{x-4 \leq i<x, y, z}, & \text { cell }_{x<i \leq x+4, y, z} \\
\text { cell }_{i, y-4 \leq j<y, z}, & \text { cell }_{i, y<j \leq y+4, z} \\
\text { cell }_{i, j, z-4 \leq k<z}, & \text { cell }_{i, j, z<k \leq z+4}
\end{array}
$$

1) Localized broadcast patterns: Dimensions $X$ and $Y$ from the grid are mapped onto the fabric of the WSE. To compute the stencil, a PE has therefore to communicate with 4 of its neighbors along each cardinal direction of the PE grid. A communication strategy similar to Rocki et al. [24], in which a single color is used per neighboring PE, would have resulted in an excessive color use for the stencil of interest to this application. In this work, localized broadcast patterns along every $\mathrm{PE}$ grid directions (Eastbound and Westbound for the $X$ dimension, and Northbound and Southbound for the $Y$ dimension) are used instead. Each broadcast pattern uses two dedicated colors (one for receiving data, one for sending data) and can happen concurrently with others broadcast patterns using separate links to communicate between PEs. Given the stencil size used in the application and the number of colors available on the hardware, the limited color usage per broadcast pattern is critical to the feasibility of the implementation.

In each broadcast pattern, multiple Root PEs send their local block of data of length $b$ to their respective neighboring 4 processing elements. This pattern is depicted in Figure $4 \mathrm{~b}$ for the Eastward direction.

The router of each PE is configured to control how packets are received and transmitted. Each router determines, for each color, the incoming links from which that color can be received and the subset of the five outgoing links to which that color will be sent. The routing can be changed at run-time by special commands which can be sent just as other packets are sent. This capability lies at the heart of the communication strategy proposed here. In Figure 4a, the different router configurations used by Finite Differences are given. All Root PEs are in configuration 0. Intermediate PEs in each broadcast are in configuration 1, while Last PEs are in configuration 2. Ideally, a Root PE should broadcast its data only to other PEs. However, due to hardware constraints, a Root PE is obliged to receive its own data as well.

After sending its local data, a Root PE sends a command to its local router and the following 4 routers. In effect, this routing update shifts the communication pattern by one position: the first neighbor now becomes a Root in the next step of the broadcast pattern. After 5 steps (and 5 shifts), a PE has sent its data out and has received data from its 4 neighbors. In Figure 4b, the target $P E$ receives data from the West during the first 4 steps, and sends its data to the East at step 5.

One of the very important aspects of this is that changing the routing on a remote router does not require any

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-05.jpg?height=570&width=1399&top_left_y=184&top_left_x=360)

Fig. 2: An overview of the Wafer Scale Engine (WSE). The WSE (to the right) occupies an entire wafer, and is a 2D array of dies. Each die is itself a grid of tiles (in the middle), which contains a router, a processing element and single-cycle access memory (to the left). In total, the WSE-2 embeds 2.6 trillion transistors in a silicon area of 46,225 $m m^{2}$.

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-05.jpg?height=474&width=483&top_left_y=1015&top_left_x=192)

(a) 3D grid of size $n x \times n y \times n z$. $X$ and $Y$ dimensions are mapped onto the PE grid of the WSE.

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-05.jpg?height=456&width=200&top_left_y=1035&top_left_x=778)

(b) Column of cells stored in local memory, extended by 8 cells. Operations are carried out by blocks of size $b$.
Fig. 3: Computing pattern mapping

action from the local PE. It is therefore uninterrupted and can perform computations simultaneously. Another advantage is that all the control logic is encapsulated in this routing update. A particular $\mathrm{PE}$ has to do two things only: sending $b$ cells out, and receiving $5 \times b$ cells (from 4 neighbors and itself). The router configuration will determine when the data flows in or out of a given router. Once a $\mathrm{PE}$ is notified that its data has been sent out, it sends a router command to update the routing and transition to the next step of the broadcast pattern. There is no bookkeeping required to determine whether a $\mathrm{PE}$ is in a given position in a broadcast.

2) Stencil computation over the $X$ and $Y$ dimensions: In order to compute the stencil over the $X$ and $Y$ axes, communications between PEs are required. As the stencil involved in this application is a 25 -point stencil, data from
16 neighboring PEs along the $X$ and $Y$ directions must be exchanged. This means that at each time step, a $\mathrm{PE}$ is involved in 4 localized broadcast patterns (one per cardinal direction). In each broadcast pattern, a PE sends its data and receives data from 4 neighbors.

Using a FMUL instruction, incoming cells from a given direction are multiplied "on the fly" with coefficients depending on their respective distance to the local cell. There are 4 FMUL operations happening concurrently (one per cardinal direction). This is depicted as step 1 in Figure 5 for the data coming from the West. Each FMUL instruction operates on $5 \times b$ incoming cells coming from a particular cardinal direction, and the coefficients (corresponding to $c_{x m}$ and $c_{y m}, \forall m \in\{1 \ldots 4\}$ in Section III). A given coefficient is applied to $b$ consecutive cells are they are coming from the same distance neighbor.

Since a $\mathrm{PE}$ is receiving data from itself, it is advantageous to compute the contribution from the center cell $x_{x, y, z}$ during this step. This is done during the FMUL operation that processes the cells coming from the West, by multiplying the cells coming from the same $\mathrm{PE}$ with $c_{x 0}+c_{y 0}+c_{z 0}$. FMULs operating on cells coming from all other directions use a coefficient of 0 for the data coming from the same PE.

Once this distributed computation phase is complete, the data of size $4 \times 5 \times b$ is reduced into a single buffer of $b$ cells (which is referred to as accumulator) using a FADD instruction (step 2 in Figure 5). The dimension of size 4 corresponds to the number of localized broadcast patterns a PE participates in, 5 corresponds to the number of $\mathrm{PE}$ it is receiving from, and $b$ is the number of elements coming from each PE. All contributions of neighboring cells along the $X$ and $Y$ dimensions are contained in the accumulator buffer after the reduction.

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-06.jpg?height=147&width=349&top_left_y=455&top_left_x=297)

(a) WSE-2 router configurations used by Finite Differences. Configuration 0 corresponds to the configuration of the Root of a broadcast, configuration 1 is used by PEs in the middle, configuration 2 is used by the Last PE.

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-06.jpg?height=688&width=894&top_left_y=171&top_left_x=908)

(b) 5 communication steps required to fetch all the data required by a target $\mathrm{PE}$ from the West (steps 1 through 4) and to send its data to the East (step 5). Corresponding router configurations are given in the circled numbers. At each step, a router command is sent through the broadcast pattern, changing the configurations of each set of 5 routers.

Fig. 4: Eastward localized broadcast operation used in Finite Differences to exchange cells along the $X$ dimension.

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-06.jpg?height=651&width=838&top_left_y=1168&top_left_x=188)

Fig. 5: A summary of main operations: computing the stencil over the $X$ and $Y$ dimensions (for each cardinal direction), reducing the accumulator buffer, and subtracting the accumulator from the wavefield.

## 3) Stencil computation over the $Z$ dimension: After

 remote contributions to the Laplacian from the $X$ and $Y$ axes of the grid have been accumulated, contributions from $Z$ can be computed. Given the problem mapping over the WSE, this means that, at each time step, the computation over the $Z$ dimension can be performed in an embarrassingly parallel fashion since this dimension resides entirely in the memory of a PE.Each PE executes 8 FMACs instructions of length $b$, multiplying the wavefield by one of the 8 coefficients

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-06.jpg?height=412&width=705&top_left_y=1168&top_left_x=1168)

(a) (c)

Fig. 6: Applying the stencil over the $Z$ dimension

(corresponding to discretization parameters $c_{z m}, \forall m \in$ $1 \ldots 4)$. The result of each FMAC is placed into the accumulator buffer (which also contains the contributions from the $X$ and $Y$ dimensions). Given a target block of size $b$ starting at coordinate $z_{b}$, each FMAC takes an input block starting at index $z_{b}+$ offset and multiplies it by a coefficient. The offset values are $\{0,1,2,3\}$ and $\{5,6,7,8\}$, and corresponding coefficients are $\left\{c_{z 4}, c_{z 3}, c_{z 2}, c_{z 1}\right\}$ and $\left\{c_{z 1}, c_{z 2}, c_{z 3}, c_{z 4}\right\}$. The CSL code is provided in Figure 7 and the first 4 steps of this process are illustrated in Figure 6. As can be seen, this step skips offset 4, which would correspond to the multiplication by $c_{z 0}$, since that particular computation has already been done as discussed earlier. At the end of this step, the Laplacian is contained in the accumulator buffer.

## D. Time integration

Once the Laplacian has been computed, the time iteration step given in Equation 2 can happen. The wavefield
accumulator $z_{z_{b} \leq i<z_{b}+b}=$ accumulator $_{z_{b} \leq i<z_{b}+b}$

$$
+z W F_{z_{b}-4 \leq i<z_{b}+b-4} \times c_{z 4}
$$

$$
\begin{aligned}
& +z W F_{z_{b}-3 \leq i<z_{b}+b-3} \times c_{z 3} \\
& +z W F_{z_{b}-2 \leq i<z_{b}+b-2} \times c_{z 2} \\
& +z W F_{z_{b}-1 \leq i<z_{b}+b-1} \times c_{z 1} \\
& +z W F_{z_{b}+1 \leq i<z_{b}+b+1} \times c_{z 1} \\
& +z W F_{z_{b}+2 \leq i<z_{b}+b+2} \times c_{z 2} \\
& +z W F_{z_{b}+3 \leq i<z_{b}+b+3} \times c_{z 3} \\
& +z W F_{z_{b}+4 \leq i<z_{b}+b+4} \times c_{z 4}
\end{aligned}
$$

(a) Operations performed along the $Z$ dimension. $z W F$ is the wavefield stored in the local memory of a $\mathrm{PE}$

```
const accumDsd = @get_dsd(mem1d_dsd, .{
    .tensor_access = |i\{nz} -> accumulator [i]});
const srcZO = @get_dsd(mem1d_dsd, . {
    .tensor_access = |i|{nz} -> zWF[0, i]});
@fmacs(accumDsd, accumDsd, srcZ0, coefficients[0]);
const srcZ1 = @increment_dsd_offset(srcZ0, 1, f32);
@fmacs(accumDsd, accumDsd, srcZ1, coefficients [1]);
const srcZ2 = @increment_dsd_offset(srcZ0, 2, f32);
@fmacs(accumDsd, accumDsd, srcZ2, coefficients [2]);
const srcZ3 = @increment_dsd_offset(srcZ0, 3, f32);
@fmacs(accumDsd, accumDsd, srcZ3, coefficients[3]);
// srcZ4 not used: update is done in Eastwards broadcast
const srcZ5 = @increment_dsd_offset(srcZ0, 5, f32);
@fmacs(accumDsd, accumDsd, srcZ5, coefficients [5]);
const srcZ6 = @increment_dsd_offset(srcZ0, 6, f32);
@fmacs(accumDsd, accumDsd, srcZ6, coefficients [6]);
const srcZ7 = @increment_dsd_offset(srcZ0, 7, f32);
```

@fmacs(accumDsd, accumDsd, srcZ7, coefficients [7]);
const srcZ8 = @increment_dsd_offset(srcZ0, 8, f32);
@fmacs(accumDsd, accumDsd, srcZ8, coefficients [8]);

(b) Equivalent CSL code. Each ๑fmacs instruction takes an output argument and three input arguments. @get_dsd returns a descriptor, corresponding to a view of an array. @increment_dsd_offset allows to offset the array pointed by an existing descriptor.

Fig. 7: Applying the stencil along the $Z$ dimension.

from the previous time step is added to the accumulator buffer. In reality, this is also done during the stencil computation: as mentioned earlier, a PE receives its own data. Doing so allows to use cycles which would have otherwise been wasted.

The next step is to update the wavefield (per Equation 2), by subtracting the wavefield to the accumulator buffer (step 3 in Figure 5).

Next, a stimulus, called source, needs to be added to a particular cell (with coordinates $(\operatorname{src} X, \operatorname{src} Y, \operatorname{src} Z)$ ) at each time step. The source value at the current time step is added to the wavefield at offset $s r c Z$ on the $\mathrm{PE}$ with coordinates ( srcX, srcY $Y$ ).

## V. Experimental Evaluation

In this section, experimental results of Finite Differences running on a Wafer-Scale Engine are presented. The scalability and energy efficiency achieved by Finite Differences on this massively parallel platform are discussed.

## A. Experimental Configuration

The experiments are conducted on two platforms: a Cerebras CS-2 equipped with a WSE-2 chip, and a GPUbased platform used as a reference. The CS-2 is Cerebras' second generation chassis, which uses the $7 \mathrm{~nm}$ WSE-2 second generation Wafer-Scale Engine. The WSE-2 offers $2.2 \times$ more processing elements than the original WSE. The experiments used a fabric of size $755 \times 994$ out of the total 850,000 processing elements of the WSE-2. The CS-2 is driven by a Linux server on which no computations take place in the context of this work. The WSE-2 platform uses Cerebras SDK 0.3.0 [27].

The GPU-based platform is Cypress from TotalEnergies. It has 4 NVIDIA A100 GPUs, each offering 40 GB of on-device RAM, a 16-core AMD EPYC 7F52 CPU, and $256 \mathrm{~GB}$ of main memory. The GPU platform is using CUDA 11.2 and GCC 8.3.1.

Numerical results produced by Finite Differences on WSE-2 are compared to the results produced by Minimod.

## B. Weak scaling Experiments

This section discusses scalability results of Finite Differences on a WSE-2 system. In order to characterize the scalability of Finite Differences, the grid dimension is modified along the $X$ and $Y$ dimensions, while the $Z$ dimension (residing in memory) is kept constant to a relevant value for this type of application. The $X$ and $Y$ dimensions are grown up to a size of $755 \times 994$. Results presented in Table II show the throughput achieved on WSE-2 in Gigacell/s, the wall-clock time required to compute 1,000 time steps on WSE-2, as well as timings on a GPGPU provided as baseline. Timing reported in this section correspond to computations taking place on the device only, be it on GPU or WSE-2.

As can be seen in the table, for all problem sizes, the wall-clock time required on WSE-2 is constant, meaning that Finite Differences scales nearly perfectly on this platform. It is crucial to observe that such a reduction in wallclock time has a significant impact in practice since this type of computations is repeated hundreds of thousands of times in an industrial context. Finite Differences reaches a throughput of 9872.78 Gcell/s on the largest problem size, which is rarely seen at single system level. This type of throughput is difficult to achieve without using a large number of nodes on distributed-memory supercomputers due to limited strong scalability.

Figure 8 depicts the ratio between the elapsed time achieved by the A100-tuned kernel compared to Finite Differences on WSE-2. As can be seen, when the largest
problem is solved (grid size of $755 \times 994 \times 1000$ ), a speedup of $228 \mathrm{x}$ is achieved. While this number shows great potential, it is understood that using multiple GPUs will likely narrow the gap. However, it is unlikely that such a performance gap can be closed entirely, given the strong scalability issues encountered by this kind of algorithm when using a large number of multi-GPU nodes in HPC clusters ( $[1],[29]$ ).

Finite Differences shows close to perfect weak scaling on WSE-2. No matter what the grid size is, the run time stays fairly stable. Taking the $200 x 200$ case as a reference, percentages of the ideal weak scaling for various grid sizes are depicted in Figure 9. As can be seen in the plot, Finite Differences systematically reaches over $98 \%$ of weak scaling efficiency. This demonstrates how extremely low latency interconnect coupled with local fast memories can be efficiently leveraged by stencil applications relying on a localized communication pattern.

In the next experiment, the sizes of the $X$ and $Y$ dimensions of the grid are fixed to $n x=755$ and $n y=994$ while the size of the $Z$ dimension $n z$ is varied from 100 to 1,000 . Results presented in Table III show that the throughput increases slightly with $n z$. This indicates that the implementation gains in efficiency due to larger block sizes $b$ and therefore lower relative overheads. More importantly, it confirms that memory accesses do not limit the performance of the implementation, confirming that it is compute-bound.

| $n x$ | $n y$ | $n z$ | Throughput <br> Gcell/s | WSE-2 <br> time $[\mathrm{s}]$ | A100 <br> time $[\mathrm{s}]$ |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 200 | 200 | 1000 | 533.64 | 0.0750 | 0.7892 |
| 400 | 400 | 1000 | 2097.60 | 0.0763 | 3.5828 |
| 600 | 600 | 1000 | 4731.53 | 0.0761 | 8.0000 |
| 755 | 500 | 1000 | 4956.17 | 0.0762 | 8.5499 |
| 755 | 600 | 1000 | 5945.40 | 0.0762 | 10.1362 |
| 755 | 900 | 1000 | 8922.08 | 0.0762 | 15.5070 |
| 755 | 990 | 1000 | 9782.14 | 0.0764 | 17.4991 |
| 755 | 994 | 1000 | 9862.78 | 0.0761 | 17.4186 |

TABLE II: Experimental results for 1,000 time steps for various grid sizes with fixed $n z$.

| $n z$ | $b$ | Throughput <br> Gcell/s | WSE-2 <br> time $[\mathrm{s}]$ | Scaling |
| :---: | :---: | :---: | :---: | :---: |
| 100 | 100 | 8688.76 | 0.8637 | 1.0000 |
| 200 | 200 | 9303.26 | 1.6133 | 1.8679 |
| 300 | 300 | 9492.89 | 2.3716 | 2.7458 |
| 400 | 400 | 9614.15 | 3.1223 | 3.6151 |
| 500 | 250 | 9786.51 | 3.8342 | 4.4392 |
| 700 | 350 | 9885.04 | 5.3143 | 6.1531 |
| 1000 | 334 | 9936.79 | 7.5524 | 8.7442 |

TABLE III: Experimental results, fixed $n x \times n y$ grid dimensions of $755 \times 994$. 100,000 time steps.

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-08.jpg?height=692&width=873&top_left_y=169&top_left_x=1079)

Fig. 8: Comparisons between implementation on WSE-2 and A100 using elapsed time describe in Table II. Fixed $n z=1000$.

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-08.jpg?height=718&width=892&top_left_y=1034&top_left_x=1077)

Fig. 9: Weak scaling under assumption that PE memory is fully utilized $(n z=1000)$.

## C. Profiling data

In the following, various profiling results are provided for Finite Differences, with the objective to provide general insights on WSE-2-based computations.

Using Cerebras' hardware profiling tool on a $600 \times$ $600 \times 1000$ grid, the execution of Finite Differences results in an average of $69.6 \%$ busy cycles. The 4 PEs at the corners of the grid have 0 busy cycles since they are not doing any computation. There is an average of $11.6 \%$ idle cycles caused by memory accesses. As expected, the load is extremely balanced, with a standard deviation of $0.8 \%$. This shows that the hardware is kept busy during the experiment, further confirming the efficiency of the
approach proposed in this work.

The power consumption of the CS-2 during a Finite Differences run on a $755 \times 994 \times 1000$ grid is reported in Figure 10. In order to record a sufficient number of samples, the run time is extended by setting the number of time steps to $10,000,000$, leading to a total run time of 754 seconds. The average power consumption during the execution is $22.8 \mathrm{~kW}$, which corresponds to 22 GFLOP/W. Such an energy efficiency is hard to find in the literature for a stencil of this order. In addition to power consumption, Figure 10 also depicts the coolant temperature of the CS-2, which uses a closed-loop water-cooling system. During the execution, the coolant temperature rises very moderately from $23.6^{\circ} \mathrm{C}$ to a peak of $25.6^{\circ} \mathrm{C}$.

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-09.jpg?height=739&width=884&top_left_y=796&top_left_x=165)

Fig. 10: Power consumption during experiment with grid $755 \times 994 \times 1000$ for $10,000,000$ time steps, which amounts to 754 seconds. In the plot, time is subsampled by 3.5 seconds. The power baseline is $16.1 \mathrm{~kW}$ and peak is 22.9 $\mathrm{kW}$. Coolant temperature is also reported, with a baseline of $23.6^{\circ} \mathrm{C}$ and a peak of $25.6^{\circ} \mathrm{C}$.

Altogether, experiments show that the Finite Differences algorithm presented in this study is able to exploit low-latency distributed memory architectures such as the WSE-2 with very high hardware utilization. The application has near-perfect weak scalability and provides significant speedups over a reference GPGPU implementation

## VI. ROOFLINE MODEL

A roofline model [33] is a synthetic view of how many floating point instructions per cycles can be done. This is the peak compute capacity of the platform. However, no computation can be done without loading data: the number of 32 bit words that can be accessed per cycle will also impact the peak compute rate. In the case of a bandwidth-bound application, the memory will actually

![](https://cdn.mathpix.com/cropped/2024_06_04_cef71d6c6d0656b6da90g-09.jpg?height=1731&width=721&top_left_y=197&top_left_x=1147)

Fig. 11: Roofline models for WSE-2 and GPGPU-based implementations (in log-log scale) for a $755 \times 994 \times 1000$ grid. Dots represent Finite Differences implementations. WSE-2 (top) has two distinct resources: memory and fabric: the leftmost blue dot corresponds to memory accesses, while the red dot to the right corresponds to fabric accesses. Rooflines are given using the same colors. The kernel is clearly in the compute-bound zone for both memory and fabric accesses.

On GPGPU (bottom), red dots and lines correspond to DRAM accesses. L1 cache accesses are depicted in blue. The kernel is clearly in the bandwidth-bound zone.

| Operation | FLOP | Mem. traffic | Fabric traffic |
| :---: | :---: | :---: | :---: |
| 17 FMUL | 1 | $1 / b$ load, 1 store | 1 load |
| 17 FADD | 1 | 2 loads, 1 store | 0 |
| 8 FMA | 2 | $2+1 /(b)$ loads, 1 store | 0 |
| 1 FSUB | 1 | 2 loads, 1 store | 0 |

TABLE IV: Instruction and memory access counts of the Finite Differences implementation on WSE-2

determine the performance of the overall application. In a similar fashion, accesses to the interconnect can be taken into account when the architecture is a distributed memory platform. This ratio between floating point operations (i.e. the "useful" operations) and the volume of data coming from/going to a given resource, for instance memory, is referred to as arithmetic intensity (measured in FLOP/byte).

The WSE-2 is a SIMD-centric multiprocessor providing up to 4 simultaneous 16 bit floating point instructions per cycle. In the context of this work, only 32 bit floating point operations are used. The peak number of floating point operations per second (FLOPs) is represented by the horizontal line at the top of Figure 11.

The WSE-2 does not have a complex cache memory hierarchy: each $\mathrm{PE}$ has a single local memory accessed directly. Four 32 bit packets can be accessed from memory per cycle, and up to two packets can be stored to memory per cycle. In memory intensive application, memory limits the peak achievable performance. This corresponds to the slanted blue line at the top of Figure 11.

Each PE is connected to its router by a bi-directional link able to move 32 bits per cycle in each direction (referred to as "off/onramp bandwidth" in the plot). The router is connected to other routers by 4 bi-directional links, each moving 32 bits per cycle. This corresponds to the slanted red line at the top of Figure 11.

For each cell, the stencil computation in Finite Differences involves 25 multiplies and 25 adds. In addition to that, Finite Differences requires a subtraction between the previous time step and the current time step. This corresponds to a total of 51 floating point instructions per cell. As explained in Section IV, due to architecture constraints, only the computation of the stencil along the $Z$ dimension can be done using FMAs. For the $X$ and $Y$ dimensions, the implementation relies on separate FMUL and FADD instructions. The final value of the current time step is computed using a FSUB operation. A summary of the instructions used in Finite Differences, floating point operations per cell, memory traffic, fabric traffic, and instruction count per cell is given in Table IV.

On WSE-2, Finite Differences computes 57 floating point operations per cell, 51 of which are required by the algorithm. Extra operations are due to hardware constraints. The number of operations strictly required by the algorithm is used in all performance numbers. These 51 floating point operations require 112 load and store of 32 bit words from/to memory, and 17 loads from fabric.
This leads to an arithmetic intensity of 0.11 with respect to memory accesses, and 0.75 with respect to fabric transfers.

On WSE-2, a $755 \times 994 \times 1000$ grid is computed in 0.0761 s (see Table II). This leads to a flop rate of 670.3 MFLOPs per PE, and an aggregated performance of 503 TFLOPs for the entire grid of PEs used by this problem size. The roofline model of the WSE-2, depicted in Figure 11(top), indicates that Finite Differences is compute bound thanks to the extremely fast local memory. This is quite remarkable, and confirms the weak scaling results given in Section V. The application is communication/memory bound on most architecture, such as the GPU platform used in this study (roofline model depicted in Figure 11(bottom)). Note that different optimizations lead to different arithmetic intensities.

## VII. ConCLUSion

In this work, a Finite Differences algorithm taking advantage of low-latency localized communications and flat memory architecture has been introduced. Localized broadcast patterns are introduced to exchange data between processing elements and fully utilize the interconnect. Experiments show that it is possible to reach near perfect weak scalability on distributed memory architecture such as the WSE-2. On this platform, the implementation of Finite Differences reaches 503 TFLOPs. This is a remarkable throughput for this stencil order on a single node machine. The roofline model introduced in this work confirms that Finite Differences becomes computebound on the WSE-2. This demonstrates the validity and potential of the approach presented in this work, and demonstrate how different hardware architectures like the WSE-2 can be exploited efficiently by stencil-based applications.

Future efforts include the integration of the ported kernel at the center of more ambitions applications, such as the ones regularly used by seismic modeling experts when taking real-life decisions. Further, given the well established capacity of this hardware architecture for MLbased applications, a hybrid HPC-ML approach will also be investigated.

One interesting consequence of having a relatively compact machine delivering such a high performance level for this type of application is that seismic data processing can happen at the same time it is acquired on the field, which is key when constant monitoring is required. Furthermore, under this scenario, processing capacity can move from data centers closer to where sensors are, namely target edge-HPC

## ACKNOWLEDGEMENTS

Authors would like to thank Cerebras Systems and TotalEnergies for allowing to share the material. Also, authors would like to acknowledge Grace Ho, Ashay Rane, and Natalia Vassilieva from Cerebras for the contributions, and Ruychi Sai from Rice U. for fruitful discussions about GPGPU optimized kernels.

## REFERENCES

[1] O. Anjum, M. Almasri, S. de Gonzalo, and W. Hwu, "An efficient gpu implementation and scaling for higher-order $3 \mathrm{~d}$ stencils," Information Sciences, vol. 586, pp. 326-343, Mar. 2022 .

[2] M. Araya-Polo, J. Cabezas, M. Hanzich, M. Pericas, F. Rubio, I. Gelado, M. Shafiq, E. Morancho, N. Navarro, E. Ayguade, J. M. Cela, and M. Valero, "Assessing accelerator-based hpc reverse time migration," IEEE Transactions on Parallel and Distributed Systems, vol. 22, no. 1, pp. 147-162, 2011.

[3] M. Araya-Polo, F. Rubio, R. de la Cruz, M. Hanzich, J. M. Cela, and D. P. Scarpazza, "3d seismic imaging through reversetime migration on homogeneous and heterogeneous multi-core processors," Scientific Programming, vol. 17, no. 1-2, pp. 185198,2009 .

[4] Cerebras, "Wafer-scale deep learning," in 2019 IEEE Hot Chips 31 Symposium (HCS). Los Alamitos, CA, USA: IEEE Computer Society, aug 2019, pp. 1-31. [Online]. Available: https://doi.org/10.1109/HOTCHIPS.2019.8875628

[5] K. Datta, S. Kamil, S. Williams, L. Oliker, J. Shalf, and K. Yelick, "Optimization and performance modeling of stencil computations on modern microprocessors," SIAM Review, vol. 51, no. 1, pp. 129-159, 2009. [Online]. Available: https://doi.org/10.1137/070693199

[6] R. de la Cruz and M. Araya-Polo, "Towards a multi-level cache performance model for 3d stencil computation," Procedia Computer Science, vol. 4, pp. $2146-2155,2011$, proceedings of the International Conference on Computational Science, ICCS 2011.

[7] R. de la Cruz and M. Araya-Polo, "Algorithm 942: Semistencil," ACM Trans. Math. Softw., vol. 40, no. 3, 2014.

[8] M. Frigo and V. Strumpen, "Cache oblivious stencil computations," in Proceedings of the 19th Annual International Conference on Supercomputing, ser. ICS '05. New York, NY, USA: Association for Computing Machinery, 2005, p. 361-366.

[9] S. Ghosh, T. Liao, H. Calandra, and B. M. Chapman, "Experiences with OpenMP, PGI, HMPP and OpenACC directives on ISO/TTI kernels," in 2012 SC Companion: High Performance Computing, Networking Storage and Analysis, Nov 2012, pp. $691-700$.

[10] F. Gürkaynak and J. Krüger, "Stx - stencil/tensor accelerator factsheet," https://www.european-processor-initiative.eu/wpcontent/uploads/2019/12/EPI-Technology-FS-STX.pdf, 2019.

[11] T. Gysi, C. Müller, O. Zinenko, S. Herhut, E. Davis, T. Wicky, O. Fuhrer, T. Hoefler, and T. Grosser, "Domain-specific multilevel ir rewriting for gpu," arXiv:2005.13014, 2020.

[12] B. Kahle and W. Hillis, The Connection Machine Model CM-1 Architecture, ser. Technical report (Thinking Machines Corporation). Thinking Machines Corporation, 1989. [Online]. Available: https://books.google.com/books?id= PCq7uAAACAAJ

[13] S. Kronawitter and C. Lengauer, "Polyhedral search space exploration in the exastencils code generator," ACM Trans. Archit. Code Optim., vol. 15, no. 4, oct 2018. [Online]. Available: https://doi.org/10.1145/3274653

[14] J. Krueger, D. Donofrio, J. Shalf, M. Mohiyuddin, S. Williams, L. Oliker, and F.-J. Pfreund, "Hardware/software co-design for energy-efficient seismic modeling," in Proceedings of 2011 International Conference for High Performance Computing, Networking, Storage and Analysis, ser. SC '11. New York, NY, USA: Association for Computing Machinery, 2011. [Online]. Available: https://doi.org/10.1145/2063384.2063482

[15] S. Lie, "Multi-million core, multi-wafer ai cluster," in 2021 IEEE Hot Chips 33 Symposium (HCS). IEEE Computer Society, 2021, pp. 1-41.

[16] M. Louboutin, M. Lange, F. Luporini, N. Kukreja, P. A. Witte, F. J. Herrmann, P. Velesko, and G. J. Gorman, "Devito (v3.1.0): an embedded domain-specific language for finite differences and geophysical exploration," Geoscientific Model Development, vol. 12, no. 3, pp. 1165-1187, 2019.

[17] K. Matsumura, H. R. Zohouri, M. Wahib, T. Endo, and S. Matsuoka, "An5d: Automated stencil framework for highdegree temporal blocking on gpus," in Proceedings of the 18th ACM/IEEE International Symposium on Code Generation and
Optimization, ser. CGO 2020. New York, NY, USA: Association for Computing Machinery, 2020, p. 199-211.

[18] J. Meng, A. Atle, H. Calandra, and M. ArayaPolo, "Minimod: A Finite Difference solver for Seismic Modeling," arXiv:2007.06048v1, Jul. 2020. [Online]. Available: https://arxiv.org/abs/2007.06048v1

[19] P. Moczo, J. Kristek, and M. Gális, The Finite-Difference Modelling of Earthquake Motions: Waves and Ruptures. Cambridge University Press, 2014.

[20] A. Nguyen, N. Satish, J. Chhugani, C. Kim, and P. Dubey, "3.5d blocking optimization for stencil computations on modern cpus and gpus," in SC'10: Proceedings of the 2010 ACM/IEEE International Conference for High Performance Computing, Networking, Storage and Analysis, 2010, pp. 1-13.

[21] A. Qawasmeh, M. R. Hugues, H. Calandra, and B. M. Chapman, "Performance portability in reverse time migration and seismic modelling via openacc," The International Journal of High Performance Computing Applications, vol. 31, no. 5, pp. 422440, 2017 .

[22] P. S. Rawat, M. Vaidya, A. Sukumaran-Rajam, A. Rountev, L. Pouchet, and P. Sadayappan, "On optimizing complex stencils on GPUs," in 2019 IEEE International Parallel and Distributed Processing Symposium (IPDPS), 2019, pp. 641-652.

[23] G. Rivera and C.-W. Tseng, "Tiling optimizations for 3d scientific computations," in Proceedings of the 2000 ACM/IEEE Conference on Supercomputing, ser. SC '00. USA: IEEE Computer Society, 2000, p. 32-es.

[24] K. Rocki, D. Van Essendelft, I. Sharapov, R. Schreiber, M. Morrison, V. Kibardin, A. Portnoy, J. F. Dietiker, M. Syamlal, and M. James, "Fast stencil-code computation on a waferscale processor," in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, 2020, pp. 1-14.

[25] R. Sai, J. Mellor-Crummey, X. Meng, M. Araya-Polo, and J. Meng, "Accelerating high-order stencils on GPUs," in 2020 IEEE/ACM Performance Modeling, Benchmarking and Simulation of High Performance Computer Systems (PMBS). Los Alamitos, CA, USA: IEEE Computer Society, nov 2020, pp. 86108 .

[26] R. Sai, J. Mellor-Crummey, X. Meng, K. Zhou, M. ArayaPolo, and J. Meng, "Accelerating high-order stencils on gpus," Concurrency and Computation: Practice and Experience, vol. e6467, 2021.

[27] J. Selig, "The cerebras software development kit: A technical overview," https://f.hubspotusercontent30.net/hubfs/8968533/ Cerebras\%20SDK\%20Technical\%20Overview\%20White\% 20Paper.pdf, 2022.

[28] D. E. Shaw, P. J. Adams, A. Azaria, J. A. Bank, B. Batson, A. Bell, M. Bergdorf, J. Bhatt, J. A. Butts, T. Correia, R. M. Dirks, R. O. Dror, M. P. Eastwood, B. Edwards, A. Even, P. Feldmann, M. Fenn, C. H. Fenton, A. Forte, J. Gagliardo, G. Gill, M. Gorlatova, B. Greskamp, J. Grossman, J. Gullingsrud, A. Harper, W. Hasenplaugh, M. Heily, B. C. Heshmat, J. Hunt, D. J. Ierardi, L. Iserovich, B. L. Jackson, N. P. Johnson, M. M. Kirk, J. L. Klepeis, J. S. Kuskin, K. M. Mackenzie, R. J. Mader, R. McGowen, A. McLaughlin, M. A. Moraes, M. H. Nasr, L. J. Nociolo, L. O'Donnell, A. Parker, J. L. Peticolas, G. Pocina, C. Predescu, T. Quan, J. K. Salmon, C. Schwink, K. S. Shim, N. Siddique, J. Spengler, T. Szalay, R. Tabladillo, R. Tartler, A. G. Taube, M. Theobald, B. Towles, W. Vick, S. C. Wang, M. Wazlowski, M. J. Weingarten, J. M. Williams, and K. A. Yuh, "Anton 3: Twenty microseconds of molecular dynamics simulation before lunch," in Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, ser. SC '21. New York, NY, USA: Association for Computing Machinery, 2021. [Online]. Available: https://doi.org/10.1145/3458817.3487397

[29] T. Shimokawabe, T. Endo, N. Onodera, and T. Aoki, "A stencil framework to realize large-scale computations beyond device memory capacity on gpu supercomputers," in 2017 IEEE International Conference on Cluster Computing (CLUSTER), 2017, pp. $525-529$.

[30] H. Stengel, J. Treibig, G. Hager, and G. Wellein, "Quantifying performance bottlenecks of stencil computations using the
execution-cache-memory model," in Proceedings of the 29th ACM on International Conference on Supercomputing, ser. ICS '15. New York, NY, USA: Association for Computing Machinery, 2015, p. 207-216. [Online]. Available: https: //doi.org/10.1145/2751205.2751240

[31] Z. team, "Zig programming language," https://ziglang.org/, 2018, (Accessed on 03/18/2022).

[32] F. Thaler, S. Moosbrugger, C. Osuna, M. Bianco, H. Vogt, A. Afanasyev, L. Mosimann, O. Fuhrer, T. C. Schulthess, and T. Hoefler, "Porting the cosmo weather model to manycore cpus," in Proceedings of the Platform for Advanced Scientific Computing Conference, ser. PASC '19. New York, NY, USA: Association for Computing Machinery, 2019. [Online]. Available: https://doi.org/10.1145/3324989.3325723

[33] S. Williams, A. Waterman, and D. Patterson, "Roofline: An insightful visual performance model for multicore architectures," Commun. ACM, vol. 52, no. 4, p. 65-76, apr 2009. [Online]. Available: https://doi.org/10.1145/1498765.1498785

[34] D. Wonnacott, "Using time skewing to eliminate idle time due to memory bandwidth and network limitations," in Proceedings 14th International Parallel and Distributed Processing Symposium. IPDPS 2000. IEEE, 2000, pp. 171-180.

[35] K. Zhang, H. Su, P. Zhang, and Y. Dou, "Data layout transformation for stencil computations using arm neon extension," in 2020 IEEE 22nd International Conference on High Performance Computing and Communications; IEEE 18th International Conference on Smart City; IEEE 6th International Conference on Data Science and Systems (HPCC/SmartCity/DSS), 2020, pp. 180-188.


[^0]:    §Equal contribution.

</end of paper 1>


