from pathlib import Path
import json,cv2,numpy as np
out=Path(Path('sandbox/video-distortion/false-alarm-current.txt').read_text());records=json.loads((out/'marked_records.json').read_text())
records=[x for x in records if x['id'] in [26410,26406,26396]];tiles=[]
def fit(image,w,h):
 canvas=np.full((h,w,3),240,np.uint8)
 if image is None or image.size==0:return canvas
 ih,iw=image.shape[:2];scale=min(w/iw,h/ih);r=cv2.resize(image,(max(1,int(iw*scale)),max(1,int(ih*scale))))
 y=(h-r.shape[0])//2;x=(w-r.shape[1])//2;canvas[y:y+r.shape[0],x:x+r.shape[1]]=r;return canvas
for a in records:
 image=cv2.imread(str(out/'originals'/f"{a['id']}_{a['cameraId']}_no_helmet.jpg"));h,w=image.shape[:2]
 objects=json.loads(a.get('detectionsJson') or '[]');tile=np.full((360,480,3),245,np.uint8)
 for i,line in enumerate([f"ID {a['id']}  cam {a['cameraId']}  head={a['topConfidence']}",a['eventTime']]):cv2.putText(tile,line,(6,20+i*22),cv2.FONT_HERSHEY_SIMPLEX,.5,(0,0,0),1,cv2.LINE_AA)
 tile[60:240,:240]=fit(image,240,180)
 if objects:
  x1,y1,x2,y2=objects[0]['bbox'];pw=x2-x1;ph=y2-y1
  crop=image[max(0,int(y1-.2*ph)):min(h,int(y1+.65*ph)),max(0,int(x1-.3*pw)):min(w,int(x2+.3*pw))]
  tile[60:350,245:475]=fit(crop,230,290)
  cv2.imwrite(str(out/f"crop_{a['id']}.jpg"),crop)
 tiles.append(tile)
for k in range(0,len(tiles),4):
 group=tiles[k:k+4]
 while len(group)<4:group.append(np.full_like(tiles[0],245))
 sheet=np.vstack([np.hstack(group[:2]),np.hstack(group[2:])]);cv2.imwrite(str(out/f'latest_detail_{k//4+1}.jpg'),sheet)
print(out,len(records))
