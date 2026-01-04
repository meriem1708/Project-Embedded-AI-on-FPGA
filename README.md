# Embedded AI on FPGA using HLS

This repository contains the SE project *Embedded AI on FPGA using High Level Synthesis*.

The objective of this project is to implement and accelerate a convolutional
neural network (CNN) for MNIST digit classification on a CPU/FPGA platform.

---

## Project overview

Three versions of the CNN were studied:
- **SW**: Software implementation on ARM Cortex-A9
- **HW_SEQ**: Hardware accelerator without parallelization
- **HW_PAR**: Hardware accelerator optimized using HLS pragmas

The project focuses on performance comparison and FPGA resource usage.

---

## Results

The CNN implementation achieves a recognition rate of approximately **97%**
after fixed-point conversion.

Hardware acceleration with parallelization significantly reduces execution
time compared to the pure software implementation.

---

## Repository structure

src/ # C source files
include/ # Headers and network weights
report/ # Project report


---

## Report

The final project report is available in the `report/` directory.

---

## Authors

- Meriem Naoui  
- Laura Boivin

## Supervisor

- M. Bilavarn

