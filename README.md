## 檔案說明
### Detect_Recognition.py
偵測pill_reciever_unsplited裡面是否有藥丸圖片，之後會使用Edge_detection.py將圖片裡的藥丸一一取出。

備註：pill_reciever_unsplited裡會接收到樹莓派拍攝藥丸的照片。
### first.h5
儲存模型參數的檔案
###  Load_Modul.py
將上述提到的first.h5儲存的模型載入，並調用預測的函數。
