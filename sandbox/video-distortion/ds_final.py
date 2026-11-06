import sys,time,pathlib
import yaml,cv2,cupy
from pyservicemaker import Pipeline,BufferRetriever,Receiver
out=pathlib.Path("/app/temp2/video-distortion");out.mkdir(exist_ok=True)
drop=bool(int(sys.argv[1]));sync=0; cfg=yaml.safe_load(open("/app/deploy/config.yaml"))
class Capture(BufferRetriever):
 def __init__(self,cid):
  super().__init__();self.cid=cid;self.n=0;self.saved=[];self.latest=None
 def consume(self,buffer):
  self.n+=1;t=buffer.extract(0).clone()
  if sync:cupy.cuda.runtime.deviceSynchronize()
  self.latest=t
  if self.n in (30,60,90,120):self.saved.append((self.n,t))
  return 1
from server.pipeline.builder import PipelineBuilder
from loguru import logger
cfg["cameras"]=[c for c in cfg["cameras"] if str(c["id"])=="1168"]
cfg["source"]["preserve_bitstream"]=True
p=Pipeline("video-integrity-candidate");recs=[]
PipelineBuilder(cfg,pathlib.Path("/app/deploy"))._add_sources(p,logger)
p.add("identity","slow",{"sleep-time":200000})
p.add("nvstreamdemux","demux");p.link("mux","slow","demux")
p.add("nvvideoconvert","conv",{"gpu-id":0,"compute-hw":1})
p.add("capsfilter","caps",{"caps":"video/x-raw(memory:NVMM), format=RGB"})
p.add("appsink","sink",{"emit-signals":True,"sync":False,"async":0})
p.link(("demux","conv"),("src_%u",""));p.link("conv","caps","sink")
rec=Capture("1168");recs.append(rec)
p.attach("sink",Receiver("capture",rec),tips="new-sample")
p.start();time.sleep(35)
for rec in recs:
 for n,t in rec.saved:
  arr=cupy.from_dlpack(t).get();arr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
  cv2.imwrite(str(out/f"final_{rec.cid}_{n}.jpg"),arr)
 print("DONE",rec.cid,sync,rec.n,len(rec.saved),flush=True)
p.stop();p.wait()
