from pathlib import Path
import json,time,cv2,yaml,os
os.environ.setdefault('OMP_NUM_THREADS','2')
from ultralytics import YOLO
root=Path('/home/pzq/projects/VideoInspector');out=Path((root/'sandbox/video-distortion/false-alarm-current.txt').read_text())
cfg=yaml.safe_load((root/'deploy/config.yaml').read_text());records=json.loads((out/'marked_records.json').read_text())
checkpoint=root/'deploy/models/helmet/yolo26s_helm_soft_hat_20261105_0925/weights/best.pt'
print('PINNED_CHECKPOINT',checkpoint,flush=True)
helmet=YOLO(str(checkpoint))
person=YOLO(str(root/'deploy'/cfg['model']['gies']['person']['source']))
print('HELMET_CLASSES',helmet.names,flush=True)
results=[]
for a in records:
 if a['alarmType']!='no_helmet':continue
 p=out/'originals'/f"{a['id']}_{a['cameraId']}_no_helmet.jpg";image=cv2.imread(str(p))
 if image is None:continue
 h=helmet.predict(image,imgsz=960,conf=.25,iou=.7,device=0,rect=False,half=True,verbose=False)[0]
 pp=person.predict(image,imgsz=640,conf=.25,iou=.7,device=0,rect=False,half=True,classes=[0],verbose=False)[0]
 hb=[{'label':h.names[int(b.cls)],'conf':float(b.conf),'bbox':b.xyxy[0].tolist()} for b in h.boxes]
 pb=[{'conf':float(b.conf),'bbox':b.xyxy[0].tolist()} for b in pp.boxes if float(b.conf)>=.6]
 original=json.loads(a.get('detectionsJson') or '[]')
 matches=[]
 for obj in original:
  x1,y1,x2,y2=obj['bbox'];hits=[]
  for b in hb:
   bx1,by1,bx2,by2=b['bbox'];cx=(bx1+bx2)/2;cy=(by1+by2)/2
   if x1<=cx<=x2 and y1<=cy<=y2:hits.append(b)
  matches.append({'original_person_bbox':obj['bbox'],'head_hits':[b for b in hits if b['label']=='head' and b['conf']>=.5],'helmet_hits':[b for b in hits if b['label']=='helmet']})
 vis=image.copy()
 for b in hb:
  x1,y1,x2,y2=map(int,b['bbox']);color=(255,0,255) if b['label']=='head' else (255,255,0)
  cv2.rectangle(vis,(x1,y1),(x2,y2),color,2);cv2.putText(vis,f"R_{b['label']} {b['conf']:.2f}",(max(0,x1),max(14,y1-5)),cv2.FONT_HERSHEY_SIMPLEX,.5,color,1,cv2.LINE_AA)
 cv2.imwrite(str(out/f"replay_{a['id']}.jpg"),vis)
 r={'checkpoint':str(checkpoint),'imgsz':960,'rect':False,'precision':'FP16','id':a['id'],'camera':a['cameraId'],'time':a['eventTime'],'original_head_conf':a['topConfidence'],'helmet_predictions':hb,'person_predictions':pb,'original_box_matches':matches}
 results.append(r)
 (out/'replay_results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
 print('REPLAY',a['id'],'head>=.5',[(round(b['conf'],3),[round(v) for v in b['bbox']]) for b in hb if b['label']=='head' and b['conf']>=.5],'helmet',sum(b['label']=='helmet' for b in hb),flush=True)
print('DONE',len(results),flush=True)
