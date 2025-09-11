bl_info = {
    "name": "Fantasy Crystal Generator",
    "author": "you + a helpful assistant",
    "version": (1, 6, 0),
    "blender": (4, 0, 0),
    "location": "View3D > N-Panel > Crystals",
    "description": "Fantasy crystal clusters with fractal branching, distributions, live preview cluster, radial growth, angle distribution, top enlargement, and custom polygon sides",
    "category": "Add Mesh",
}

import bpy
import bmesh
from mathutils import Vector, Euler, Matrix
from math import radians, pi, cos, sin
import random, time

# =========================
# Material
# =========================

def ensure_material(name="FantasyCrystal"):
    mat = bpy.data.materials.get(name)
    if mat:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nt = mat.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)

    out = nt.nodes.new("ShaderNodeOutputMaterial"); out.location = (800, 0)
    principled = nt.nodes.new("ShaderNodeBsdfPrincipled"); principled.location = (0, 0)
    principled.inputs["Transmission"].default_value = 1.0
    principled.inputs["Roughness"].default_value = 0.02
    principled.inputs["IOR"].default_value = 1.45

    emission = nt.nodes.new("ShaderNodeEmission"); emission.location = (0, -250)
    emission.inputs["Strength"].default_value = 0.0

    add_shader = nt.nodes.new("ShaderNodeAddShader"); add_shader.location = (300, -100)

    hue = nt.nodes.new("ShaderNodeHueSaturation"); hue.location = (-250, 0)
    hue.inputs["Saturation"].default_value = 1.0
    hue.inputs["Value"].default_value = 1.0

    color_ramp = nt.nodes.new("ShaderNodeValToRGB"); color_ramp.location = (-500, 0)
    color_ramp.color_ramp.elements[0].position = 0.15
    color_ramp.color_ramp.elements[0].color = (0.6, 0.1, 1.0, 1.0)
    color_ramp.color_ramp.elements[1].position = 0.85
    color_ramp.color_ramp.elements[1].color = (0.0, 1.0, 1.0, 1.0)

    obj_info = nt.nodes.new("ShaderNodeObjectInfo"); obj_info.location = (-750, 0)

    tex_noise = nt.nodes.new("ShaderNodeTexNoise"); tex_noise.location = (-250, -250)
    tex_noise.inputs["Scale"].default_value = 20.0
    tex_noise.inputs["Detail"].default_value = 8.0
    bump = nt.nodes.new("ShaderNodeBump"); bump.location = (150, -300)
    bump.inputs["Strength"].default_value = 0.1

    nt.links.new(obj_info.outputs["Random"], color_ramp.inputs["Fac"])
    nt.links.new(color_ramp.outputs["Color"], hue.inputs["Color"])
    nt.links.new(hue.outputs["Color"], principled.inputs["Base Color"])
    nt.links.new(hue.outputs["Color"], emission.inputs["Color"])
    nt.links.new(tex_noise.outputs["Fac"], bump.inputs["Height"])
    nt.links.new(bump.outputs["Normal"], principled.inputs["Normal"])
    nt.links.new(principled.outputs["BSDF"], add_shader.inputs[0])
    nt.links.new(emission.outputs["Emission"], add_shader.inputs[1])
    nt.links.new(add_shader.outputs["Shader"], out.inputs["Surface"])
    return mat

def set_material_params(mat, roughness, ior, emission):
    nt = mat.node_tree
    for n in nt.nodes:
        if n.type == 'BSDF_PRINCIPLED':
            n.inputs["Roughness"].default_value = roughness
            n.inputs["IOR"].default_value = ior
        if n.type == 'EMISSION':
            n.inputs["Strength"].default_value = emission

# =========================
# Modifiers / mesh utils
# =========================

def add_displace_modifier(obj, strength=0.06, midlevel=0.0, scale=2.0, detail=6):
    # Legacy texture datablock used by Displace modifier
    tex = bpy.data.textures.new(name=f"DispTex_{obj.name}", type='MUSGRAVE')
    tex.noise_scale = float(scale)
    if hasattr(tex, "dimension"):  tex.dimension  = 2.0
    if hasattr(tex, "lacunarity"): tex.lacunarity = 2.0
    if hasattr(tex, "octaves"):    tex.octaves    = max(1, int(detail))
    mod = obj.modifiers.new("CrystalDisplace", 'DISPLACE')
    mod.texture   = tex
    mod.strength  = float(strength)
    mod.mid_level = float(midlevel)
    mod.direction = 'NORMAL'

def add_bevel_modifier(obj, width=0.01, segments=2, angle=radians(30)):
    mod = obj.modifiers.new("CrystalBevel", 'BEVEL')
    mod.width = width
    mod.segments = segments
    mod.limit_method = 'ANGLE'
    mod.angle_limit = angle

def shade_smooth_autosmooth(obj, angle=radians(30)):
    mesh = obj.data
    mesh.use_auto_smooth = True
    mesh.auto_smooth_angle = angle
    for p in mesh.polygons:
        p.use_smooth = True

def apply_object_scale(obj):
    # Operator-free: bake object scale into mesh and reset scale.
    if not hasattr(obj.data, "transform"):
        return
    sx, sy, sz = obj.scale
    M = Matrix.Diagonal((sx, sy, sz, 1.0))
    obj.data.transform(M)
    obj.scale = (1.0, 1.0, 1.0)

# =========================
# Sampling helpers
# =========================

def sample_pdf(u, pdf="UNIFORM"):
    if pdf == "UNIFORM":
        return u
    if pdf == "BELL":
        # bell-ish curve around 0.5
        return (3*u*u - 2*u*u*u + (1 - (3*(1-u)*(1-u) - 2*(1-u)*(1-u)*(1-u)))) * 0.5
    if pdf == "SKEW_SMALL":
        return u*u           # bias small
    if pdf == "SKEW_LARGE":
        return u**0.5        # bias large
    return u

def apply_center_bias(value01, r_norm, center_bias):
    """
    center_bias: -1..+1 (positive => bigger near center, negative => bigger near edge)
    r_norm: 0 center, 1 edge (relative to cluster radius)
    """
    influence = (1.0 - r_norm) if center_bias >= 0 else r_norm
    k = abs(center_bias)
    target = 1.0 if center_bias > 0 else 0.0
    return (1 - k*influence) * value01 + (k*influence) * target

# =========================
# Crystal geometry
# =========================

def make_prism(height=2.0, radius=0.2, sides=6, taper_top=0.5, top_enlarge=1.0,
               tipify=True, tip_height_ratio=0.12):
    """
    General N-gon prism with tapered (or flared) and optionally enlarged top, with a tip.
    Returns an unlinked object (caller links to target collection).
    """
    sides = max(3, int(sides))
    mesh = bpy.data.meshes.new("CrystalMesh")
    obj = bpy.data.objects.new("Crystal", mesh)

    bm = bmesh.new()
    bmesh.ops.create_circle(bm, cap_ends=True, radius=radius, segments=sides)

    # Extrude up
    geom_extrude = bmesh.ops.extrude_face_region(bm, geom=[f for f in bm.faces])
    verts_extruded = [ele for ele in geom_extrude["geom"] if isinstance(ele, bmesh.types.BMVert)]
    for v in verts_extruded:
        v.co.z += height

    faces_sorted = sorted(bm.faces, key=lambda f: f.calc_center_median().z)
    top_face = faces_sorted[-1]

    # Taper/flare
    top_center = top_face.calc_center_median()
    factor = max(0.001, taper_top)
    for v in top_face.verts:
        v.co.xy = top_center.xy + (v.co.xy - top_center.xy) * factor

    # Additional enlargement (post-taper), for big cap forms
    if top_enlarge != 1.0:
        for v in top_face.verts:
            v.co.xy = top_center.xy + (v.co.xy - top_center.xy) * max(0.001, float(top_enlarge))

    # Tip
    if tipify and tip_height_ratio > 0:
        tip_h = height * tip_height_ratio
        geom_extrude2 = bmesh.ops.extrude_face_region(bm, geom=[top_face])
        v2 = [ele for ele in geom_extrude2["geom"] if isinstance(ele, bmesh.types.BMVert)]
        for v in v2:
            v.co.z += tip_h
        new_faces = [ele for ele in geom_extrude2["geom"] if isinstance(ele, bmesh.types.BMFace)]
        if new_faces:
            top2 = new_faces[0]
            top2_center = top2.calc_center_median()
            for v in top2.verts:
                v.co.xy = top2_center.xy + (v.co.xy - top2_center.xy) * 0.2

    bm.to_mesh(mesh); bm.free()
    return obj

# =========================
# Params container
# =========================

class CRYSTAL_Params:
    def __init__(self, props):
        self.count = int(props.count)
        self.seed  = int(props.seed)
        self.cluster_radius = float(props.cluster_radius)
        self.inner_spawn_radius = max(0.0, min(float(props.inner_spawn_radius), self.cluster_radius))
        # ranges
        self.min_height = float(props.min_height)
        self.max_height = float(props.max_height)
        self.base_radius_min = float(min(props.base_radius_min, props.base_radius_max))
        self.base_radius_max = float(max(props.base_radius_min, props.base_radius_max))
        # shape
        self.sides = int(props.sides)
        self.taper_top = float(props.taper_top)          # 0..2
        self.top_enlarge = float(props.top_enlarge)      # independent post-taper scale
        self.tip_ratio = float(props.tip_ratio)
        self.lean_bias_deg = float(props.lean_bias_deg)  # can be negative
        self.max_lean_deg  = float(props.max_lean_deg)
        # surface
        self.disp_strength  = float(props.disp_strength)
        self.disp_scale     = float(props.disp_scale)
        self.disp_detail    = int(props.disp_detail)
        self.bevel_width    = float(props.bevel_width)
        self.bevel_segments = int(props.bevel_segments)
        self.bevel_angle_deg= float(props.bevel_angle_deg)
        # material
        self.roughness = float(props.roughness)
        self.ior       = float(props.ior)
        self.emission  = float(props.emission)
        # fractal
        self.fractal_depth       = int(props.fractal_depth)
        self.fractal_branch_prob = float(props.fractal_branch_prob)
        self.fractal_scale_min   = float(min(props.fractal_scale_min, props.fractal_scale_max))
        self.fractal_scale_max   = float(min(0.95, max(props.fractal_scale_min, props.fractal_scale_max)))  # < 1.0
        self.fractal_branch_tilt = float(props.fractal_branch_tilt)
        # distributions
        self.size_pdf = props.size_pdf
        self.center_size_bias = float(props.center_size_bias)
        # preview
        self.preview_use_full = bool(props.preview_use_full)
        self.preview_count = int(props.preview_count)
        # radial growth
        self.radial_growth = bool(props.radial_growth)
        self.radial_tilt_mode = props.radial_tilt_mode   # "CONSTANT" or "CENTER_TO_EDGE"
        self.surface_tilt_deg = float(props.surface_tilt_deg)
        self.angle_center_deg = float(props.angle_center_deg)
        self.angle_edge_deg   = float(props.angle_edge_deg)
        self.surface_tilt_jitter = float(props.surface_tilt_jitter)
        self.radial_yaw_random = bool(props.radial_yaw_random)

# =========================
# Generation helpers
# =========================

def sample_height_and_radius(params, r_norm):
    u_h = sample_pdf(random.random(), params.size_pdf)
    u_r = sample_pdf(random.random(), params.size_pdf)
    u_h = apply_center_bias(u_h, r_norm, params.center_size_bias)
    u_r = apply_center_bias(u_r, r_norm, params.center_size_bias)
    height = params.min_height + u_h * (params.max_height - params.min_height)
    base_r = params.base_radius_min + u_r * (params.base_radius_max - params.base_radius_min)
    return height, base_r

def pick_position(inner_r, outer_r):
    # uniform area sampling in an annulus
    r = (random.random()*(outer_r**2 - inner_r**2) + inner_r**2) ** 0.5
    t = random.uniform(0, 2*pi)
    return r, t, Vector((r * cos(t), r * sin(t), 0.0))

def lean_euler(lean_bias_deg, max_lean_deg):
    # symmetric lean around bias; negatives allowed
    x = random.uniform(-max_lean_deg, max_lean_deg) + lean_bias_deg
    y = random.uniform(-max_lean_deg, max_lean_deg) + lean_bias_deg
    z = random.uniform(0, 360)
    return Euler((radians(x), radians(y), radians(z)), 'XYZ')

def align_z_to_vector(vec: Vector):
    # Return a rotation aligning +Z to 'vec'
    if vec.length == 0:
        return Euler((0.0, 0.0, 0.0), 'XYZ')
    v = vec.normalized()
    quat = v.to_track_quat('Z', 'Y')  # track +Z to v
    return quat.to_euler()

def crystal_one(height, base_radius, params):
    obj = make_prism(
        height=height,
        radius=base_radius * random.uniform(0.9, 1.1),
        sides=params.sides,
        taper_top=params.taper_top,
        top_enlarge=params.top_enlarge,
        tipify=True,
        tip_height_ratio=params.tip_ratio,
    )
    obj.rotation_euler = lean_euler(params.lean_bias_deg, params.max_lean_deg)
    obj.scale = (random.uniform(0.9,1.1), random.uniform(0.9,1.1), random.uniform(0.95,1.05))
    apply_object_scale(obj)
    add_displace_modifier(obj, params.disp_strength, 0.0, params.disp_scale, params.disp_detail)
    add_bevel_modifier(obj, params.bevel_width, params.bevel_segments, radians(params.bevel_angle_deg))
    shade_smooth_autosmooth(obj, radians(30))
    mat = ensure_material("FantasyCrystal")
    if obj.data.materials: obj.data.materials[0] = mat
    else: obj.data.materials.append(mat)
    set_material_params(mat, params.roughness, params.ior, params.emission)
    return obj

def branch_children(parent, parent_height, parent_radius, params, depth_left, collection):
    if depth_left <= 0:
        return
    n_branches = random.randint(1, 4)
    for _ in range(n_branches):
        if random.random() > params.fractal_branch_prob:
            continue
        # strictly thinner/smaller than parent
        scale = random.uniform(params.fractal_scale_min, params.fractal_scale_max)  # < 1
        c_height = max(0.05, parent_height * scale)
        c_radius = max(0.01, parent_radius * scale)

        child = crystal_one(c_height, c_radius, params)
        # place near parent tip
        top_local = Vector((0, 0, parent.dimensions.z * 0.6))
        child.location = parent.matrix_world @ top_local
        # orient child with tilt
        child.rotation_euler = Euler((
            radians(random.uniform(-params.fractal_branch_tilt, params.fractal_branch_tilt)),
            radians(random.uniform(-params.fractal_branch_tilt, params.fractal_branch_tilt)),
            radians(random.uniform(0, 360))
        ), 'XYZ')
        apply_object_scale(child)
        collection.objects.link(child)
        branch_children(child, c_height, c_radius, params, depth_left - 1, collection)

# =========================
# Build cluster
# =========================

def build_cluster(params, collection, count=None, with_base=True):
    created = []
    total = params.count if count is None else int(count)
    denom = max(1e-8, (params.cluster_radius - params.inner_spawn_radius))

    for _ in range(total):
        r, t, pos = pick_position(params.inner_spawn_radius, params.cluster_radius)
        r_norm = 0.0 if params.cluster_radius <= 1e-8 else min(1.0, r / params.cluster_radius)  # 0 center .. 1 edge of whole cluster
        r_annulus = min(1.0, max(0.0, (r - params.inner_spawn_radius) / denom))  # 0 at inner_spawn_radius .. 1 at cluster_radius

        height, base_r = sample_height_and_radius(params, r_norm)
        c = crystal_one(height, base_r, params)
        c.location = pos

        # Orientation
        if params.radial_growth:
            # Align +Z to outward radial normal
            eul_align = align_z_to_vector(pos)
            c.rotation_euler = eul_align

            # Base tilt: constant OR center->edge interpolation
            if params.radial_tilt_mode == "CENTER_TO_EDGE":
                base_tilt = params.angle_center_deg + r_annulus * (params.angle_edge_deg - params.angle_center_deg)
            else:
                base_tilt = params.surface_tilt_deg

            # Jitter and optional yaw around normal
            tilt = base_tilt + random.uniform(-params.surface_tilt_jitter, params.surface_tilt_jitter)
            if params.radial_yaw_random:
                c.rotation_euler.rotate_axis('Z', radians(random.uniform(0, 360)))
            # Tilt away from normal around a tangent axis (local X)
            c.rotation_euler.rotate_axis('X', radians(tilt))
        # else: keep lean-based random euler from crystal_one()

        collection.objects.link(c)
        if params.fractal_depth > 0:
            branch_children(c, height, base_r, params, params.fractal_depth, collection)
        created.append(c)

    base = None
    if with_base:
        try:
            bpy.ops.mesh.primitive_ico_sphere_add(subdivisions=3, radius=params.cluster_radius*0.65, location=(0,0,-0.25))
            base = bpy.context.active_object
            base.name = f"ClusterBase_{params.seed}"
            add_displace_modifier(base, strength=0.15, scale=3.5, detail=5)
            rock_mat = bpy.data.materials.get("CrystalBaseRock") or bpy.data.materials.new("CrystalBaseRock")
            rock_mat.use_nodes = True
            bsdf = rock_mat.node_tree.nodes.get("Principled BSDF")
            if bsdf:
                bsdf.inputs["Base Color"].default_value = (0.05,0.05,0.05,1.0)
                bsdf.inputs["Roughness"].default_value = 1.0
                bsdf.inputs["Specular"].default_value = 0.1
            if base.data.materials: base.data.materials[0] = rock_mat
            else: base.data.materials.append(rock_mat)
            collection.objects.link(base)
        except:
            pass
    return created, base

# =========================
# Live preview cluster
# =========================

_preview_guard = False

def _ensure_collection(name):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

def _clear_collection(coll):
    for obj in list(coll.objects):
        try:
            bpy.data.objects.remove(obj, do_unlink=True)
        except:
            pass

def _clear_preview_now():
    coll = bpy.data.collections.get("CrystalPreview_Collection")
    if coll:
        _clear_collection(coll)

def _build_preview(props, context):
    global _preview_guard
    if _preview_guard or not props.live_preview:
        return
    _preview_guard = True
    try:
        random.seed(props.seed)
        coll = _ensure_collection("CrystalPreview_Collection")
        _clear_collection(coll)
        p = CRYSTAL_Params(props)

        if getattr(props, "preview_use_full", True):
            preview_n = max(1, int(p.count))
        else:
            preview_n = max(1, min(int(props.preview_count), int(p.count)))

        build_cluster(p, coll, count=preview_n, with_base=False)
    finally:
        _preview_guard = False

def _update_callback(self, context):
    _build_preview(self, context)

# =========================
# UI Properties
# =========================

class CRYSTALGEN_Props(bpy.types.PropertyGroup):
    # counts/seed
    count: bpy.props.IntProperty(name="Count", default=25, min=1, max=100000, update=_update_callback)
    seed:  bpy.props.IntProperty(name="Seed",  default=12345, min=0, update=_update_callback)

    # cluster & ranges
    cluster_radius: bpy.props.FloatProperty(name="Cluster Radius", default=1.6, min=0.0, update=_update_callback)
    inner_spawn_radius: bpy.props.FloatProperty(name="Inner Spawn Radius", default=0.0, min=0.0, update=_update_callback)
    min_height: bpy.props.FloatProperty(name="Min Height", default=1.2, min=0.05, update=_update_callback)
    max_height: bpy.props.FloatProperty(name="Max Height", default=2.8, min=0.1, update=_update_callback)

    base_radius_min: bpy.props.FloatProperty(name="Base Radius Min", default=0.10, min=0.01, update=_update_callback)
    base_radius_max: bpy.props.FloatProperty(name="Base Radius Max", default=0.18, min=0.01, update=_update_callback)

    # shape
    sides: bpy.props.IntProperty(name="Sides (3–24)", default=6, min=3, max=24, update=_update_callback)
    taper_top: bpy.props.FloatProperty(name="Top Taper (0–2)", default=0.45, min=0.0, max=2.0, update=_update_callback)
    top_enlarge: bpy.props.FloatProperty(name="Top Enlarge (×)", default=1.0, min=0.1, max=3.0, update=_update_callback)
    tip_ratio:  bpy.props.FloatProperty(name="Tip Height Ratio", default=0.12, min=0.0, max=0.4, update=_update_callback)
    lean_bias_deg: bpy.props.FloatProperty(name="Lean Bias (deg)", default=0.0, min=-45.0, max=45.0, update=_update_callback)
    max_lean_deg:  bpy.props.FloatProperty(name="Max Lean Span (deg)", default=8.0, min=0.0, max=45.0, update=_update_callback)

    # surface
    disp_strength: bpy.props.FloatProperty(name="Chip/Displace Strength", default=0.06, min=0.0, max=0.5, update=_update_callback)
    disp_scale:    bpy.props.FloatProperty(name="Displace Scale", default=2.0, min=0.1, max=10.0, update=_update_callback)
    disp_detail:   bpy.props.IntProperty(  name="Displace Detail (octaves)", default=6, min=1, max=12, update=_update_callback)

    bevel_width:    bpy.props.FloatProperty(name="Bevel Width", default=0.01, min=0.0, max=0.2, update=_update_callback)
    bevel_segments: bpy.props.IntProperty(  name="Bevel Segments", default=2, min=1, max=6, update=_update_callback)
    bevel_angle_deg:bpy.props.FloatProperty(name="Bevel Limit Angle", default=30.0, min=1.0, max=60.0, update=_update_callback)

    # material
    roughness: bpy.props.FloatProperty(name="Surface Roughness", default=0.03, min=0.0, max=1.0, update=_update_callback)
    ior:       bpy.props.FloatProperty(name="IOR", default=1.45, min=1.0, max=2.5, update=_update_callback)
    emission:  bpy.props.FloatProperty(name="Glow (Emission Strength)", default=0.0, min=0.0, max=10.0, update=_update_callback)

    # fractal
    fractal_depth:       bpy.props.IntProperty(  name="Fractal Depth", default=0, min=0, max=3, update=_update_callback)
    fractal_branch_prob: bpy.props.FloatProperty(name="Branch Probability", default=0.6, min=0.0, max=1.0, update=_update_callback)
    fractal_scale_min:   bpy.props.FloatProperty(name="Branch Scale Min", default=0.35, min=0.05, max=0.95, update=_update_callback)
    fractal_scale_max:   bpy.props.FloatProperty(name="Branch Scale Max", default=0.6,  min=0.05, max=0.95, update=_update_callback)
    fractal_branch_tilt: bpy.props.FloatProperty(name="Branch Tilt (deg)", default=18.0, min=0.0, max=60.0, update=_update_callback)

    # distributions
    size_pdf: bpy.props.EnumProperty(
        name="Size PDF",
        items=[("UNIFORM","Uniform","Flat"),
               ("BELL","Bell","More mid-sized"),
               ("SKEW_SMALL","Skew Small","Bias toward smaller"),
               ("SKEW_LARGE","Skew Large","Bias toward larger")],
        default="UNIFORM", update=_update_callback
    )
    center_size_bias: bpy.props.FloatProperty(
        name="Center Size Bias (-1 edge big … +1 center big)",
        default=0.5, min=-1.0, max=1.0, update=_update_callback
    )

    # radial growth
    radial_growth: bpy.props.BoolProperty(
        name="Radial Growth (Spokes)",
        default=False,
        description="Align crystals to the outward normal; spawn in annulus",
        update=_update_callback
    )
    radial_tilt_mode: bpy.props.EnumProperty(
        name="Tilt Mode",
        items=[("CONSTANT","Constant","Use a single tilt angle"),
               ("CENTER_TO_EDGE","Center→Edge","Interpolate angle by radius")],
        default="CENTER_TO_EDGE", update=_update_callback
    )
    surface_tilt_deg: bpy.props.FloatProperty(
        name="Constant Tilt (deg)",
        default=10.0, min=0.0, max=90.0, update=_update_callback
    )
    angle_center_deg: bpy.props.FloatProperty(
        name="Angle at Center (deg)",
        default=0.0, min=0.0, max=90.0, update=_update_callback
    )
    angle_edge_deg: bpy.props.FloatProperty(
        name="Angle at Edge (deg)",
        default=18.0, min=0.0, max=90.0, update=_update_callback
    )
    surface_tilt_jitter: bpy.props.FloatProperty(
        name="Tilt Jitter (deg)",
        default=5.0, min=0.0, max=90.0, update=_update_callback
    )
    radial_yaw_random: bpy.props.BoolProperty(
        name="Random Yaw Around Normal",
        default=True, update=_update_callback
    )

    # live preview
    live_preview: bpy.props.BoolProperty(
        name="Live Preview (Cluster)",
        default=False,
        description="Regenerate a preview cluster on any change",
        update=_update_callback
    )
    preview_use_full: bpy.props.BoolProperty(
        name="Preview Uses Full Count",
        default=True,
        description="If on, preview builds exactly 'Count'; if off, uses Preview Limit",
        update=_update_callback
    )
    preview_count: bpy.props.IntProperty(
        name="Preview Limit",
        default=12, min=1, max=100000,
        update=_update_callback
    )

# =========================
# Operators
# =========================

def _make_collection(name):
    coll = bpy.data.collections.get(name)
    if not coll:
        coll = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(coll)
    return coll

class CRYSTALGEN_OT_generate_random(bpy.types.Operator):
    """Generate a fully random cluster within reasonable limits"""
    bl_idname = "crystalgen.generate_random_cluster"
    bl_label = "Generate Crystal Cluster"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Clear preview, pause live preview
        try:
            _clear_preview_now()
        except:
            pass
        props = getattr(context.scene, "crystalgen", None)
        restore_live = None
        if props:
            restore_live = props.live_preview
            props.live_preview = False

        seed = int(time.time()*1000) & 0x7fffffff
        random.seed(seed)
        class P: pass
        p = P()
        # randomized, sane ranges
        p.count = random.randint(14, 42)
        p.seed  = seed
        p.cluster_radius = random.uniform(1.0, 2.0)
        p.inner_spawn_radius = random.uniform(0.0, p.cluster_radius*0.4)
        p.min_height = random.uniform(0.8, 1.6)
        p.max_height = max(p.min_height + 0.4, random.uniform(2.0, 3.2))
        p.base_radius_min = random.uniform(0.08, 0.12)
        p.base_radius_max = random.uniform(0.13, 0.20)
        p.sides = random.choice([5,6,6,7,8])
        p.taper_top = random.uniform(0.35, 1.6)
        p.top_enlarge = random.uniform(1.0, 1.6)
        p.tip_ratio = random.uniform(0.08, 0.16)
        p.lean_bias_deg = random.uniform(-6.0, 6.0)
        p.max_lean_deg  = random.uniform(4.0, 15.0)
        p.disp_strength = random.uniform(0.04, 0.12)
        p.disp_scale    = random.uniform(1.4, 3.0)
        p.disp_detail   = random.randint(5, 8)
        p.bevel_width    = random.uniform(0.006, 0.018)
        p.bevel_segments = random.randint(1, 3)
        p.bevel_angle_deg= random.uniform(25.0, 45.0)
        p.roughness = random.uniform(0.01, 0.06)
        p.ior       = random.uniform(1.38, 1.55)
        p.emission  = random.uniform(0.0, 2.5)
        # fractal (sometimes)
        p.fractal_depth       = random.choice([0,0,1,1,2])
        p.fractal_branch_prob = random.uniform(0.4, 0.75)
        p.fractal_scale_min   = 0.3
        p.fractal_scale_max   = 0.65
        p.fractal_branch_tilt = random.uniform(12.0, 24.0)
        # distributions
        p.size_pdf = random.choice(["UNIFORM","BELL","SKEW_SMALL","SKEW_LARGE"])
        p.center_size_bias = random.uniform(-0.6, 0.8)
        # radial
        p.radial_growth = random.choice([False, False, True])
        p.radial_tilt_mode = random.choice(["CENTER_TO_EDGE","CONSTANT"])
        p.surface_tilt_deg = random.uniform(6.0, 18.0)
        p.angle_center_deg = random.uniform(0.0, 6.0)
        p.angle_edge_deg   = random.uniform(10.0, 24.0)
        p.surface_tilt_jitter = random.uniform(2.0, 10.0)
        p.radial_yaw_random = True
        # preview
        p.preview_use_full = True
        p.preview_count = 12

        params = CRYSTAL_Params(p)
        coll = _make_collection(f"CrystalCluster_{seed}")
        created, _ = build_cluster(params, coll, with_base=True)

        if props and restore_live is not None:
            props.live_preview = restore_live

        self.report({'INFO'}, f"Random cluster (seed {seed}) with {len(created)} roots in {coll.name}")
        return {'FINISHED'}

class CRYSTALGEN_OT_generate_custom(bpy.types.Operator):
    """Generate a cluster using the panel settings (seeded)"""
    bl_idname = "crystalgen.generate_custom"
    bl_label = "Generate Custom"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        # Clear preview, pause live preview
        try:
            _clear_preview_now()
        except:
            pass
        props = context.scene.crystalgen
        restore_live = props.live_preview
        props.live_preview = False

        random.seed(props.seed)
        params = CRYSTAL_Params(props)
        coll = _make_collection(f"CrystalCluster_{props.seed}")
        created, _ = build_cluster(params, coll, with_base=True)

        props.live_preview = restore_live

        self.report({'INFO'}, f"Custom cluster (seed {props.seed}) with {len(created)} roots in {coll.name}")
        return {'FINISHED'}

# =========================
# Panel
# =========================

class CRYSTALGEN_PT_panel(bpy.types.Panel):
    bl_label = "Fantasy Crystals"
    bl_idname = "CRYSTALGEN_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = 'Crystals'

    def draw(self, context):
        props = context.scene.crystalgen
        layout = self.layout

        row = layout.row(align=True)
        row.operator("crystalgen.generate_random_cluster", text="Generate Crystal Cluster")
        row.operator("crystalgen.generate_custom", text="Generate Custom")

        layout.separator()
        box = layout.box()
        box.prop(props, "live_preview")
        row = box.row(align=True)
        row.prop(props, "preview_use_full")
        if not props.preview_use_full:
            row = box.row(align=True)
            row.prop(props, "preview_count")

        col = layout.box().column(align=True)
        col.label(text="Counts, Seed & Cluster")
        col.prop(props, "count")
        col.prop(props, "seed")
        col.prop(props, "cluster_radius")
        col.prop(props, "inner_spawn_radius")

        col = layout.box().column(align=True)
        col.label(text="Size Ranges")
        col.prop(props, "min_height"); col.prop(props, "max_height")
        row = col.row(align=True)
        row.prop(props, "base_radius_min"); row.prop(props, "base_radius_max")

        col = layout.box().column(align=True)
        col.label(text="Distributions")
        col.prop(props, "size_pdf")
        col.prop(props, "center_size_bias")

        col = layout.box().column(align=True)
        col.label(text="Shape")
        col.prop(props, "sides")
        col.prop(props, "taper_top")
        col.prop(props, "top_enlarge")
        col.prop(props, "tip_ratio")
        col.prop(props, "lean_bias_deg")
        col.prop(props, "max_lean_deg")

        col = layout.box().column(align=True)
        col.label(text="Surface Detail")
        col.prop(props, "disp_strength")
        col.prop(props, "disp_scale")
        col.prop(props, "disp_detail")
        col.prop(props, "bevel_width")
        col.prop(props, "bevel_segments")
        col.prop(props, "bevel_angle_deg")

        col = layout.box().column(align=True)
        col.label(text="Material")
        col.prop(props, "roughness")
        col.prop(props, "ior")
        col.prop(props, "emission")

        col = layout.box().column(align=True)
        col.label(text="Fractal Branching")
        col.prop(props, "fractal_depth")
        col.prop(props, "fractal_branch_prob")
        col.prop(props, "fractal_scale_min")
        col.prop(props, "fractal_scale_max")
        col.prop(props, "fractal_branch_tilt")

        col = layout.box().column(align=True)
        col.label(text="Radial Growth (Spokes)")
        col.prop(props, "radial_growth")
        col.prop(props, "radial_tilt_mode")
        if props.radial_tilt_mode == "CENTER_TO_EDGE":
            row = col.row(align=True)
            row.prop(props, "angle_center_deg")
            row.prop(props, "angle_edge_deg")
        else:
            col.prop(props, "surface_tilt_deg")
        col.prop(props, "surface_tilt_jitter")
        col.prop(props, "radial_yaw_random")

# =========================
# Register
# =========================

classes = (
    CRYSTALGEN_Props,
    CRYSTALGEN_OT_generate_random,
    CRYSTALGEN_OT_generate_custom,
    CRYSTALGEN_PT_panel,
)

def register():
    for c in classes:
        bpy.utils.register_class(c)
    bpy.types.Scene.crystalgen = bpy.props.PointerProperty(type=CRYSTALGEN_Props)

def unregister():
    del bpy.types.Scene.crystalgen
    for c in reversed(classes):
        bpy.utils.unregister_class(c)

if __name__ == "__main__":
    register()
