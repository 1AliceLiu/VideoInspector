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
import gi
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GObject
Gst.init(None)
_base=type(Gst.ElementFactory.make("nvurisrcbin"))
class ProtectedSource(_base):
 __gtype_name__="VideoIntegritySource"
 __gstmetadata__=("Video Integrity Source","Source/Video","Suppress flagged corrupt output","VideoInspector")
 def __init__(self):
  super().__init__()
  self.connect("child-added",self.on_child)
 def on_child(self,proxy,child,name):
  try:
   if GObject.signal_lookup("child-added",child.__gtype__):child.connect("child-added",self.on_child)
   if child.find_property("discard-corrupted-frames"):
    child.set_property("discard-corrupted-frames",True)
    child.set_property("automatic-request-sync-points",True)
    print("PROTECTED_DECODER",name,flush=True)
  except Exception as e:print("PROTECT_ERROR",type(e).__name__,str(e),flush=True)
GObject.type_register(ProtectedSource)
assert Gst.Element.register(None,"protectedurisrcbin",Gst.Rank.NONE,ProtectedSource)

p=Pipeline("distortion-protected");recs=[]
p.add("nvstreammux","mux",{"batch-size":1,"width":1920,"height":1080,"batched-push-timeout":33000,"live-source":1})
p.add("identity","slow",{"sleep-time":200000});p.add("nvstreamdemux","demux");p.link("mux","slow","demux")
for idx,cid in enumerate(("1168",)):
 uri=next(c["rtsp_url"] for c in cfg["cameras"] if str(c["id"])==cid)
 p.add("protectedurisrcbin",f"src{idx}",{"uri":uri,"select-rtp-protocol":4,"drop-frame-interval":2,"latency":500,"drop-on-latency":drop,"max-size-buffers":4})
 p.add("queue",f"raw{idx}",{"max-size-buffers":2,"max-size-bytes":0,"max-size-time":0,"leaky":2});p.link(f"src{idx}",f"raw{idx}");p.link((f"raw{idx}","mux"),("","sink_%u"))
 p.add("queue",f"q{idx}",{"max-size-buffers":4,"leaky":2})
 p.add("nvvideoconvert",f"conv{idx}",{"gpu-id":0,"compute-hw":1})
 p.add("capsfilter",f"caps{idx}",{"caps":"video/x-raw(memory:NVMM), format=RGB"})
 p.add("appsink",f"sink{idx}",{"emit-signals":True,"sync":False,"async":0})
 p.link(("demux",f"q{idx}"),("src_%u",""));p.link(f"q{idx}",f"conv{idx}",f"caps{idx}",f"sink{idx}")
 rec=Capture(cid);recs.append(rec)
 p.attach(f"sink{idx}",Receiver(f"capture{idx}",rec),tips="new-sample")
p.start();time.sleep(20)
for rec in recs:
 for n,t in rec.saved:
  arr=cupy.from_dlpack(t).get();arr=cv2.cvtColor(arr,cv2.COLOR_RGB2BGR)
  cv2.imwrite(str(out/f"protected_{rec.cid}_{n}.jpg"),arr)
 print("DONE",rec.cid,sync,rec.n,len(rec.saved),flush=True)
p.stop();p.wait()
