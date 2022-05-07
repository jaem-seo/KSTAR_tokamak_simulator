# KSTAR tokamak simulator (KSTAR-NN)
- KSTAR is a tokamak (donut-shaped nuclear fusion device) located in South Korea.
- This repository provides a KSTAR tokamak simulation tool with LSTM-based neural network.
- See also [AI Tokamak Control](https://github.com/jaem-seo/AI_tokamak_control) where the AI replaces the manual control in this simulator.

# Installation
- You can install by
```
$ git clone https://github.com/jaem-seo/KSTAR_tokamak_simulator.git
$ cd KSTAR_tokamak_simulator
```

# Try it out
- Open the GUI by typing below. It might take a bit depending on your environment.
```
$ python kstar_simulator_v0.py
```
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/165520027-c4f79698-a816-49a3-8e75-fd44985ad95c.png">
</p>

- Slide the toggles in the left side and see the fusion plasma evolution in the right side.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/165654530-c8230a8c-e9a7-4574-bae3-bab646bb61dc.gif">
</p>

- I hope you get insight with this virtual experiment!

# Note
- This simulation has been tested with many real discharges, and shows acceptable prediction accuracy.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/165522817-bc56771f-600b-4c7c-a9c3-4da0256bfe3e.png">
</p>

- But it does not always guarantee perfect prediction since it doesn't account for all unknown factors.
- For example, the experiments #18672 and #22671 were conducted under almost the same setting, but showed quite different behaviors.
- In this case, the simulation shows quite a reasonable, average prediction as shown below.
<p align="center">
  <img src="https://user-images.githubusercontent.com/46472432/165521918-bd6969bf-31e0-4bf8-8848-f6ee6afeefaa.png">
</p>


# License
```
MIT License

Copyright (c) 2022 Jaemin Seo

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
© 2022 GitHub, Inc.
Terms
```

# References
- Seo, Jaemin, et al. "Feedforward beta control in the KSTAR tokamak by deep reinforcement learning." Nuclear Fusion [61.10 (2021): 106010.](https://iopscience.iop.org/article/10.1088/1741-4326/ac121b/meta)
- Seo, Jaemin, et al. Nuclear Fusion (2022) (In review).
