"""Closest convex OBB features for witness-only finite road edge contact.

At a certified *clear* instant, the nearest points of two disjoint closed
boxes belong to vertex-face or edge-edge pairs. We enumerate every vertex
projection and every pair of 12 box edges, never infer an infinite road plane.
"""
from __future__ import annotations
from dataclasses import dataclass
from math import isfinite
from .body_collision import _box_axes
from .geometry.vector import Vec3
from .track.volume import OrientedBoxVolume

@dataclass(frozen=True,slots=True)
class BoxFeaturePair:
    body_point:Vec3
    obstacle_point:Vec3
    separation_m:float
    feature:str
    version:str="obb_closest_features_v1"

def corners(box:OrientedBoxVolume):
    u=_box_axes(box)
    e=box.half_extents
    out=[]
    for i in range(8):
        signs=tuple(1. if i&(1<<j) else -1. for j in range(3))
        p=box.center
        for axis,half,sign in zip(u,(e.x,e.y,e.z),signs,strict=True):
            p=p+axis*(half*sign)
        out.append(p)
    return tuple(out)

def closest_point(box:OrientedBoxVolume,p:Vec3)->Vec3:
    u=_box_axes(box)
    e=box.half_extents
    q=box.center
    delta=p-q
    for axis,half in zip(u,(e.x,e.y,e.z),strict=True):
        x=max(-half,min(half,delta.dot(axis)))
        q=q+axis*x
    return q

def edges(points):
    return tuple((points[i],points[i^(1<<j)]) for i in range(8)
                 for j in range(3) if not i&(1<<j))

def _segment_pair(a:Vec3,b:Vec3,c:Vec3,d:Vec3):
    # Closest points on two finite segments (Ericson), with parallel guards.
    u,v,w=b-a,d-c,a-c
    aa,bb,cc,dd,ee=u.dot(u),u.dot(v),v.dot(v),u.dot(w),v.dot(w)
    if aa<=0. or cc<=0.:raise ValueError("degenerate rigid box edge")
    det=aa*cc-bb*bb
    sn,sd,tn,td=det,det,det,det
    if det<1e-16*aa*cc:
        sn=0.;sd=1.;tn=ee;td=cc
    else:
        sn=bb*ee-cc*dd;tn=aa*ee-bb*dd
        if sn<0.:
            sn=0.;tn=ee;td=cc
        elif sn>sd:
            sn=sd;tn=ee+bb;td=cc
    if tn<0.:
        tn=0.
        if -dd<0.:sn=0.
        elif -dd>aa:sn=sd
        else:sn=-dd;sd=aa
    elif tn>td:
        tn=td
        if -dd+bb<0.:sn=0.
        elif -dd+bb>aa:sn=sd
        else:sn=-dd+bb;sd=aa
    s=sn/sd if abs(sn)>0. else 0.
    t=tn/td if abs(tn)>0. else 0.
    return a+u*s,c+v*t

def closest_obb_features(body:OrientedBoxVolume,obstacle:OrientedBoxVolume)->BoxFeaturePair:
    if not isinstance(body,OrientedBoxVolume) or not isinstance(obstacle,OrientedBoxVolume):
        raise ValueError("two registered OBBs required")
    b,o=corners(body),corners(obstacle)
    candidates=[]
    for v in b:
        candidates.append(((v-closest_point(obstacle,v)).magnitude,v,
                           closest_point(obstacle,v),"body_vertex_to_road"))
    for v in o:
        candidates.append(((closest_point(body,v)-v).magnitude,
                           closest_point(body,v),v,"road_vertex_to_body"))
    for a,b2 in edges(b):
        for c,d in edges(o):
            p,q=_segment_pair(a,b2,c,d)
            candidates.append(((p-q).magnitude,p,q,"edge_to_edge"))
    best=min(candidates,key=lambda row:row[0])
    if not isfinite(best[0]) or best[0]<=0.:
        raise ValueError("boxes intersect/touch: cannot certify distinct clear feature pair")
    return BoxFeaturePair(best[1],best[2],best[0],best[3])
