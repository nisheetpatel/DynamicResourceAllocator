# DynamicResourceAllocator (DRA)
Code for "Nisheet Patel, Luigi Acerbi, and Alexandre Pouget. Dynamic allocation of limited memory resources in reinforcement learning. _NeurIPS_ 2020."

## Note
I am aware that this code is nightmarish to navigate through. I have it on my to-do list to clean it up, but it may take a while. In the meantime, please contact me if you'd like to implement any part of it and I'll be happy to share the new, updated code for the gradient-free or gradient-based resource allocators :)

## Usage
Clone the repository to a folder of your choice and run the files to reproduce the relevant figures in the paper. For using DRA with any of the environments from OpenAI gym, refer to the file ./scripts/dynamicResourceAllocator.py. For custom environments such as the 2D gridworld, refer to the files ./scripts/\*\_fig1\*.py, which reproduce the figures in the code and also provide additional examples of the implementation.

## Summary of the work
![NeurIPS2020_poster](/figures/NeurIPS_poster_final.png)
[Here](https://github.com/nisheetpatel/DynamicResourceAllocator/blob/master/figures/NeurIPS_poster_final.pdf) is the link to the high-resolution pdf poster.
