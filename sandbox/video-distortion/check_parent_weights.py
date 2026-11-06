from pathlib import Path
import json,cv2
from ultralytics import YOLO
out=Path(Path('sandbox/video-distortion/false-alarm-current.txt').read_text());records={a['id']:a for a in json.loads((out/'marked_records.json').read_text())}
p=Path('/home/pzq/projects/helmet/runs/detect/yolo26s_helm_20261029_0656/weights/best.pt')
if not p.exists():raise SystemExit('parent checkpoint missing')
model=YOLO(str(p));rows=[]
for aid in (26329,26328,26323,26320):
 a=records[aid];image=cv2.imread(str(out/'originals'/f"{aid}_{a['cameraId']}_no_helmet.jpg"));bbox=json.loads(a['detectionsJson'])[0]['bbox']
 pred=model.predict(image,imgsz=960,rect=False,conf=.25,iou=.7,device=0,half=True,verbose=False)[0];hits=[]
 for b in pred.boxes:
  x1,y1,x2,y2=b.xyxy[0].tolist();cx=(x1+x2)/2;cy=(y1+y2)/2
  if bbox[0]<=cx<=bbox[2] and bbox[1]<=cy<=bbox[3]:hits.append({'label':pred.names[int(b.cls)],'confidence':float(b.conf),'bbox':[x1,y1,x2,y2]})
 rows.append({'id':aid,'parent_imgsz':960,'hits':hits});print(aid,[(h['label'],round(h['confidence'],3)) for h in hits],flush=True)
(out/'parent_checkpoint_comparison.json').write_text(json.dumps(rows,indent=2))
