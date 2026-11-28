
import NXOpen
import NXOpen.Features
import json

def modify_expr(expr_name: str, value: float, session: NXOpen.Session, unit: str = None):
    part = session.Parts.Work
    expr = part.Expressions.FindObject(expr_name)
    unit = NXOpen.Unit.Null if unit is None else part.UnitCollection.FindObject(unit)
    part.Expressions.EditExpressionWithUnits(expr, unit, f"{value:.6f}")

data_file = "D:/Doc/fastopt/runtime/data.json"

with open(data_file, "r") as fin:
    data = json.load(fin)

session = NXOpen.Session.GetSession()
part = session.Parts.Work
markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "modify airfoils")

for name, coords in zip(data["afs_name"], data["afs"]):
    spline = part.Features.FindObject(name)
    splineEx = part.Features.CreateStudioSplineBuilderEx(spline)
    for idx,i in enumerate(coords):
         geoCon = splineEx.ConstraintManager.FindItem(idx)
         pt = geoCon.Point
         pt.SetCoordinates(NXOpen.Point3d(-i[1]*1000.0, 0.0, i[0]*1000.0))
    
    splineEx.Evaluate()
    spline= splineEx.Commit()
    splineEx.Destroy()

session.UpdateManager.DoUpdate(markid)

markid = session.SetUndoMark(NXOpen.Session.MarkVisibility.Visible, "modify expr")
expr_list=data["expr"]
for expr_name,(expr_val,expr_unit)in expr_list.items():
    modify_expr(expr_name, expr_val, session, expr_unit)
session.UpdateManager.DoUpdate(markid)
