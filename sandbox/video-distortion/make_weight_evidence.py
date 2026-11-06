from pathlib import Path
import json,cv2,numpy as np
out=Path(Path('sandbox/video-distortion/false-alarm-current.txt').read_text());records={a['id']:a for a in json.loads((out/'marked_records.json').read_text())};data=json.loads((out/'controlled_size_comparison.json').read_text());rows=[]
for aid in (26329,26328,26320):
 a=records[aid];original=cv2.imread(str(out/'originals'/f"{aid}_{a['cameraId']}_no_helmet.jpg"));h,w=original.shape[:2];x1,y1,x2,y2=json.loads(a['detectionsJson'])[0]['bbox'];pw=x2-x1;ph=y2-y1
 cols=[]
 for name in ('old','new'):
  image=original.copy();hits=next(r['hits'] for r in data if r['id']==aid and r['weights']==name and r['imgsz']==960)
  for b in hits:
   xx1,yy1,xx2,yy2=map(int,b['bbox']);color=(255,0,255) if b['label']=='head' else (255,255,0);cv2.rectangle(image,(xx1,yy1),(xx2,yy2),color,3)
  crop=image[max(0,int(y1-.12*ph)):min(h,int(y1+.65*ph)),max(0,int(x1-.2*pw)):min(w,int(x2+.2*pw))]
  canvas=np.full((300,400,3),245,np.uint8);ch,cw=crop.shape[:2];scale=min(390/cw,255/ch);res=cv2.resize(crop,(max(1,int(cw*scale)),max(1,int(ch*scale))));tx=(400-res.shape[1])//2;canvas[40:40+res.shape[0],tx:tx+res.shape[1]]=res
  msg=name.upper()+' @960: '+(', '.join(f"{b['label']} {b['confidence']:.3f}" for b in hits) or 'no matched detection')
  cv2.putText(canvas,f'ID {aid} '+msg,(5,23),cv2.FONT_HERSHEY_SIMPLEX,.42,(0,0,0),1,cv2.LINE_AA);cols.append(canvas)
 rows.append(np.hstack(cols))
cv2.imwrite(str(out/'weight_evidence.jpg'),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,85])
print(out/'weight_evidence.jpg')
