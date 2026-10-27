from fastapi import File, UploadFile

class SinglePredictionDto:
    def __init__(self, file: UploadFile = File(...)):
        self.file = file