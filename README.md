## Running the Code
- Create a virtual environment
```
sudo apt-get install python3-venv
python3 -m venv myvenv
source myvenv/bin/activate
```
- Install required packages
- We followed the same experimental setting with 
- "Hsu et al., "What Makes Graph Neural Networks Miscalibrated?" (NeurIPS'22)

- Install following packages
```
pip install torch==1.7.0+cu110 torchvision==0.8.0+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.0.6 torch-sparse==0.6.8 -f https://data.pyg.org/whl/torch-1.7.0+cu110.html
```
```
pip install -r requirements.txt
```
- To reproduce the calibration performance of Simi-Mailbox on CoraFull (GAT), run the following script:
```
python main_calibration_small.py
```
