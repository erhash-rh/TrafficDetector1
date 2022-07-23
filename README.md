# TrafficDetector1

A YOLO-like network which detects and tracks 3 classes of vehicles.

Run:
```Python
inference.py
```

The model was trained on the Google Open Images dataset for the respective classes. 

The network architecture can be inspected in:
```Python
model.py
```

The tracker implements a variation of the Hungarian algorithm.