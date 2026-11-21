
import NXOpen
import NXOpen.Features
import json

data_file = "D:/Doc/fastopt/runtime/data.json"

with open(data_file, "r") as fin:
    data = json.load(fin)

session = NXOpen.Session.GetSession()
part = session.Parts.Work
markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "create airfoils")

for name, coords in zip(data["afs_name"], data["afs"]):
    splineEX = part.Features.CreateStudioSplineBuilderEx(NXOpen.NXObject.Null)
    for i in coords:
        pt = NXOpen.Point3d(i[1]*-1000.0, 0.0, i[0] * 1000.0)
        pt = part.Points.CreatePoint(pt)
        gcons = splineEX.ConstraintManager.CreateGeometricConstraintData()
        gcons.Point = pt
        splineEX.ConstraintManager.Append(gcons)
    spline= splineEX.Commit()
    spline.SetName(name)
    splineEX.Destroy()

session.UpdateManager.DoUpdate(markid)

