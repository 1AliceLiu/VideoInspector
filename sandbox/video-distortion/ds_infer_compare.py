import sys,time,pathlib,threading,json
import yaml,cv2,cupy
from pyservicemaker import Pipeline,BufferRetriever,Receiver
out=pathlib.Path("/app/temp2/video-distortion");out.mkdir(exist_ok=True)
mode=int(sys.argv[1]); sync=int(sys.argv[2]) if len(sys.argv)>2 else 0
cfg=yaml.safe_load(open("/app/deploy/config.yaml"))
url=next(c["rtsp_url"] for c in cfg["cameras"] if str(c["id"])=="1168")
class Capture(BufferRetriever):
 def __init__(self):
  super().__init__();self.n=0;self.saved=[];self.latest=None
 def consume(self,buffer):
  self.n+=1
  t=buffer.extract(0).clone()
  self.latest=t
  if self.n in (30,60,90,120):
   if sync: cupy.cuda.runtime.deviceSynchronize()
   self.saved.append((self.n,t))
   print("CAPTURE",mode,sync,self.n,t.shape,t.strides,flush=True)
  return 1
rec=Capture();p=Pipeline("distortion-test")
p.add("nvurisrcbin","src",{"uri":url,"select-rtp-protocol":4,"drop-frame-interval":mode,"latency":500})
p.add("nvstreammux","mux",{"batch-size":1,"width":1920,"height":1080,"batched-push-timeout":33000,"live-source":1})
p.link(("src","mux"),("","sink_%u"))
from server.pipeline.builder import PipelineBuilder
from loguru import logger
builder=PipelineBuilder({**cfg,"cameras":[next(c for c in cfg["cameras"] if str(c["id"])=="1168")]},pathlib.Path("/app/deploy"))
builder._add_inference_chain(p)
p.add("nvvideoconvert","conv",{"gpu-id":0,"compute-hw":1})
p.add("capsfilter","caps",{"caps":"video/x-raw(memory:NVMM), format=RGB"})
p.add("appsink","sink",{"emit-signals":True,"sync":False,"async":0})
p.link(("demux","conv"),("src_%u",""));p.link("conv","caps","sink")
p.attach("sink",Receiver("capture",rec),tips="new-sample")
p.start();time.sleep(20)
for n,t in rec.saved:
 arr=cupy.from_dlpack(t).get();arr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
 cv2.imwrite(str(out/f"infer_drop{mode}_sync{sync}_{n}.jpg"),arr)
print("DONE",mode,sync,rec.n,len(rec.saved),flush=True)
p.stop();p.wait()
