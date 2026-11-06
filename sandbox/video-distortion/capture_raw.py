import os,time,pathlib,json
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"]="rtsp_transport;tcp"
import cv2,yaml
out=pathlib.Path("/home/pzq/projects/VideoInspector/sandbox/video-distortion")
cfg=yaml.safe_load(open("/home/pzq/projects/VideoInspector/deploy/config.yaml"))
url=next(c["rtsp_url"] for c in cfg["cameras"] if str(c["id"])=="1168")
cap=cv2.VideoCapture(url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,8000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,8000])
print("OPEN",cap.isOpened(),flush=True)
t=time.monotonic();n=0
while cap.isOpened() and time.monotonic()-t<15:
 ok,f=cap.read()
 if not ok:break
 n+=1
 if n in (1,30,60,120):
  p=out/f"raw_tcp_{n}.jpg";cv2.imwrite(str(p),f);print("FRAME",n,f.shape,flush=True)
cap.release();print("FRAMES",n,flush=True)
