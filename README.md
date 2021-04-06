# Object Detection Pi

For edgetpu:

```sh
echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
```

Install this library:

```sh
sudo apt-get install libedgetpu1-std
```
Install virtualenv:

```sh
sudo pip3 install virtualenv
```

Make virtualenv:

```sh
python3 -m venv web-obj-detection-env
```

Activate virtualenv:

```sh
source web-obj-detection-env/bin/activate
```

Install python libraries:

```sh
pip install -r requirements.txt
```

Run the program w/o edgetpu:

```sh
python3 webstream.py
```

Run the program with edgetpu:

```sh
python3 webstream --edgetpu
```
