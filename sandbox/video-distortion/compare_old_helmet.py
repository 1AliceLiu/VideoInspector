from pathlib import Path
import json,cv2
from ultralytics import YOLO
out=Path(Path('sandbox/video-distortion/false-alarm-current.txt').read_text());records=json.loads((out/'marked_records.json').read_text());new={r['id']:r for r in json.loads((out/'replay_results.json').read_text())}
p=Path('deploy/models/helmet/yolo26s_helm_20260804_0844/weights/best.pt');model=YOLO(str(p));results=[]
for a in records:
 if a['id'] not in new:continue
 image=cv2.imread(str(out/'originals'/f"{a['id']}_{a['cameraId']}_no_helmet.jpg"))
 pred=model.predict(image,imgsz=640,rect=False,conf=.25,iou=.7,half=True,device=0,verbose=False)[0]
 boxes=[{'label':pred.names[int(b.cls)],'conf':float(b.conf),'bbox':b.xyxy[0].tolist()} for b in pred.boxes]
 matches=[]
 for m in new[a['id']]['original_box_matches']:
  x1,y1,x2,y2=m['original_person_bbox'];hits=[]
  for b in boxes:
   xx1,yy1,xx2,yy2=b['bbox'];cx=(xx1+xx2)/2;cy=(yy1+yy2)/2
   if x1<=cx<=x2 and y1<=cy<=y2:hits.append(b)
  matches.append({'bbox':[x1,y1,x2,y2],'old_hits':hits,'new_head_hits':m['head_hits'],'new_helmet_hits':m['helmet_hits']})
 r={'id':a['id'],'old_predictions':boxes,'matches':matches};results.append(r)
 (out/'old_new_comparison.json').write_text(json.dumps(results,indent=2))
 old_head=max([b['conf'] for m in matches for b in m['old_hits'] if b['label']=='head'] or [0]);new_head=max([b['conf'] for m in matches for b in m['new_head_hits']] or [0])
 old_helmet=max([b['conf'] for m in matches for b in m['old_hits'] if b['label']=='helmet'] or [0]);new_helmet=max([b['conf'] for m in matches for b in m['new_helmet_hits']] or [0])
 print(a['id'],'old head/helmet',round(old_head,3),round(old_helmet,3),'new head/helmet',round(new_head,3),round(new_helmet,3),flush=True)
print('DONE',len(results),flush=True)
