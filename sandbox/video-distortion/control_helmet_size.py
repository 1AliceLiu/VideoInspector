from pathlib import Path
import json,cv2,yaml
from ultralytics import YOLO
out=Path(Path('sandbox/video-distortion/false-alarm-current.txt').read_text());records={a['id']:a for a in json.loads((out/'marked_records.json').read_text())}
paths={'old':'deploy/models/helmet/yolo26s_helm_20260804_0844/weights/best.pt','new':yaml.safe_load(Path('deploy/config.yaml').read_text())['model']['gies']['helmet']['source']}
paths['new']='deploy/'+paths['new'];rows=[]
for name,path in paths.items():
 model=YOLO(path)
 for size in (640,960):
  for aid in (26329,26328,26323,26320,26384):
   a=records[aid];image=cv2.imread(str(out/'originals'/f"{aid}_{a['cameraId']}_no_helmet.jpg"));dets=json.loads(a['detectionsJson']);b0=dets[0]['bbox']
   pred=model.predict(image,imgsz=size,rect=False,conf=.25,iou=.7,device=0,half=True,verbose=False)[0];hits=[]
   for b in pred.boxes:
    x1,y1,x2,y2=b.xyxy[0].tolist();cx=(x1+x2)/2;cy=(y1+y2)/2
    if b0[0]<=cx<=b0[2] and b0[1]<=cy<=b0[3]:hits.append({'label':pred.names[int(b.cls)],'confidence':float(b.conf),'bbox':[x1,y1,x2,y2]})
   row={'id':aid,'weights':name,'imgsz':size,'hits':hits};rows.append(row)
   print(name,size,aid,[(b['label'],round(b['confidence'],3)) for b in hits],flush=True)
(out/'controlled_size_comparison.json').write_text(json.dumps(rows,indent=2))
