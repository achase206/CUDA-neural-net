# Project 2 - Implementing Neural Network in CUDA

In this repository is a code a code that "discovers" the Morse potential via a feedforward neural network and backpropagation.
You'll now work towards getting the code running efficiently on GPUs.
As you'll note, this `README.md` doesn't include much specific information on exactly what you should do.
Use the skills and knowledge you've gained over the course of this semester to make intelligent decisions wherever the instructions leave room for interpretation.

## Task 1 - Port the Code to GPUs

The code currently runs on the CPU.
Rewrite it to run on GPUs, making an effort optimize for efficiency.

## Task 2 - Profile the Code

Perform profiling tests on Perlmutter, including analysis of Roofline plots.
Analyze the time cost of training your model with respect to the number of hidden layers, the size of the hidden layers, and the size of the training dataset.
Include your data and plots here, and explain your conclusions regarding the profiling results.

## Task 3 - Increase the Size of the Calculation

Note that the Morse potential problem is too small to effectively utilize Perlmutter's resources.
Modify your code to solve a more physically complex problem that can better utilize the Perlmutter GPUs.
For example, what if instead of trying to learn a Morse potential for a two-body system, you tried to learn a potential energy surface for a three-body or four-body system?
You may select a physical problem unrelated to molecular dynamics, if you prefer.
Provide this code **in addition** to the code for Tasks 1 and 2; in other words, submit code that solves the new problem as well as code that discovers the Morse potential.

Repeat your Perlmutter profiling calculations with the new system and discuss your results.

## Task 4 - Discuss the Code

Discuss your code's parallelization strategy.
Why did you choose this strategy?
In what ways could the code's performance be improved?
Describe some ways in which your neural network implementation would need to change to accomodate machine learning in the context of a condensed-phase molecular dynamics simulation involving thousands of atoms.

Your response to this task should be fairly extensive (>1,000 words).

## Answers

