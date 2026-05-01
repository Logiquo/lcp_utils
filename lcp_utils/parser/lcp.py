from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from warnings import warn
from xml.etree import ElementTree as ET

X_NS = "adobe:ns:meta/"
RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
PHOTOSHOP_NS = "http://ns.adobe.com/photoshop/1.0/"
ST_CAMERA_NS = "http://ns.adobe.com/photoshop/1.0/camera-profile"
DEFAULT_XMP_TOOLKIT = (
    "Adobe XMP Core 7.0-c000 1.000000, 0000/00/00-00:00:00        "
)

NS = {
    "x": X_NS,
    "rdf": RDF_NS,
    "photoshop": PHOTOSHOP_NS,
    "stCamera": ST_CAMERA_NS,
}

ET.register_namespace("x", X_NS)
ET.register_namespace("rdf", RDF_NS)
ET.register_namespace("photoshop", PHOTOSHOP_NS)
ET.register_namespace("stCamera", ST_CAMERA_NS)


@dataclass(kw_only=True)
class Vignette:
    """Vignetting correction model.

    Attributes:
        focal_length_x: Horizontal focal length normalized by the reference
            image's longer dimension. It anchors the brightness falloff model
            to the same coordinate system as the geometric correction.
        focal_length_y: Vertical focal length normalized by the reference
            image's longer dimension.
        image_x_center: Horizontal center of the vignette pattern in normalized
            image coordinates. When omitted by a profile, consumers commonly
            use the image center.
        image_y_center: Vertical center of the vignette pattern in normalized
            image coordinates. When omitted by a profile, consumers commonly
            use the image center.
        param1: Required first coefficient of the radial brightness falloff
            polynomial.
        param2: Second coefficient of the radial brightness falloff polynomial.
        param3: Third coefficient of the radial brightness falloff polynomial.
        residual_mean_error: Expected average percentage prediction error for
            the fitted vignette model.
        piecewise_params: Use picecwise function instead of polinomial for the
            vignette correction.
    """

    focal_length_x: float | None = None
    focal_length_y: float | None = None
    image_x_center: float | None = None
    image_y_center: float | None = None
    param1: float
    param2: float | None = None
    param3: float | None = None
    residual_mean_error: float | None = None
    piecewise_params: list[tuple[float, float]] = field(default_factory=list)


@dataclass(kw_only=True)
class Perspective:
    """Rectilinear geometric correction model.

    Adobe Lightroom no longer relies on lens profiles for lateral chromatic
    aberration correction, so this class intentionally describes geometric
    distortion only.

    Attributes:
        version: Version of the rectilinear model definition. Defaults to 2,
            which is the version used by the bundled samples.
        focal_length_x: Horizontal focal length normalized by the reference
            image's longer dimension. Consumers scale it by the target image's
            longer dimension before applying the model.
        focal_length_y: Vertical focal length normalized by the reference
            image's longer dimension.
        image_x_center: Principal point x-coordinate normalized by the
            reference image's longer dimension.
        image_y_center: Principal point y-coordinate normalized by the
            reference image's longer dimension.
        scale_factor: Extra scale applied after correction. If a profile omits
            it, the specification's neutral value is 1.0.
        radial_distort_param1: Required first radial distortion coefficient.
        radial_distort_param2: Second radial distortion coefficient.
        radial_distort_param3: Third radial distortion coefficient.
        tangential_distort_param1: First tangential distortion coefficient.
        tangential_distort_param2: Second tangential distortion coefficient.
        residual_mean_error: Expected average relative prediction error per
            pixel for the fitted geometric model.
        residual_standard_deviation: Expected spread of the relative prediction
            error per pixel.
        vignette: Optional vignetting model calibrated for the same focal
            length, aperture, and focus distance.
    """

    version: int = 2
    focal_length_x: float | None = None
    focal_length_y: float | None = None
    image_x_center: float | None = None
    image_y_center: float | None = None
    scale_factor: float | None = None
    radial_distort_param1: float
    radial_distort_param2: float | None = None
    radial_distort_param3: float | None = None
    tangential_distort_param1: float | None = None
    tangential_distort_param2: float | None = None
    residual_mean_error: float | None = None
    residual_standard_deviation: float | None = None
    vignette: Vignette | None = None


@dataclass(kw_only=True)
class Fisheye:
    """Fisheye geometric correction model.

    Adobe Lightroom no longer relies on lens profiles for lateral chromatic
    aberration correction, so this class intentionally describes geometric
    distortion only.

    Attributes:
        version: Version of the fisheye model definition. Defaults to 2 to
            match the rectilinear profiles bundled with this package.
        focal_length_x: Horizontal focal length normalized by the reference
            image's longer dimension. Consumers scale it by the target image's
            longer dimension before applying the model.
        focal_length_y: Vertical focal length normalized by the reference
            image's longer dimension.
        image_x_center: Principal point x-coordinate normalized by the
            reference image's longer dimension.
        image_y_center: Principal point y-coordinate normalized by the
            reference image's longer dimension.
        radial_distort_param1: Required first fisheye radial distortion
            coefficient.
        radial_distort_param2: Second fisheye radial distortion coefficient.
        residual_mean_error: Expected average relative prediction error per
            pixel for the fitted geometric model.
        residual_standard_deviation: Expected spread of the relative prediction
            error per pixel.
        vignette: Optional vignetting model calibrated for the same focal
            length, aperture, and focus distance.
    """

    version: int = 2
    focal_length_x: float | None = None
    focal_length_y: float | None = None
    image_x_center: float | None = None
    image_y_center: float | None = None
    radial_distort_param1: float
    radial_distort_param2: float | None = None
    residual_mean_error: float | None = None
    residual_standard_deviation: float | None = None
    vignette: Vignette | None = None


@dataclass(kw_only=True)
class Profile:
    """One calibrated camera/lens shooting condition in an LCP file.

    Attributes:
        author: Creator or organization credited for the profile.
        make: Required camera manufacturer used for automatic profile matching.
        model: Camera body model used for automatic profile matching.
        unique_camera_model: Locale-independent camera model identifier,
            typically from DNG metadata.
        camera_pretty_name: Required human-readable camera body name shown in
            profile selection UI.
        lens: Lens model identifier used for automatic profile matching.
        lens_info: Encoded lens range information, usually min/max focal
            lengths and f-numbers.
        lens_id: Camera-system-specific lens identifier when available.
        lens_pretty_name: Required human-readable lens name shown in profile
            selection UI.
        profile_name: Required human-readable profile name.
        image_width: Width in pixels of the reference image set.
        image_length: Height in pixels of the reference image set.
        x_resolution: Horizontal resolution of the reference images in DPI.
        y_resolution: Vertical resolution of the reference images in DPI.
        focal_length: Required focal length in millimeters for this
            sub-profile.
        aperture_value: Required aperture in APEX units for this sub-profile.
        camera_raw_profile: Required flag indicating whether the profile was
            built for raw images rather than rendered JPEG/TIFF input.
        focus_distance: Required average focus distance in meters for the
            reference image set.
        sensor_format_factor: Relative sensor size used to match profiles
            across camera formats.
        prefer_metadata_distort: Indicate EXIF metadata is preferred over this
            lens profile.
        perspective: Rectilinear geometric distortion model, when the lens is
            represented as a perspective lens.
        fisheye: Fisheye geometric distortion model, when the lens is
            represented as a fisheye lens.
    """

    author: str | None = None
    make: str
    model: str | None = None
    unique_camera_model: str | None = None
    camera_pretty_name: str
    lens: str | None = None
    lens_info: str | None = None
    lens_id: str | None = None
    lens_pretty_name: str
    profile_name: str
    image_width: int | None = None
    image_length: int | None = None
    x_resolution: float | None = None
    y_resolution: float | None = None
    focal_length: float
    aperture_value: float
    camera_raw_profile: bool
    focus_distance: float
    sensor_format_factor: float | None = None
    prefer_metadata_distort: bool | None = None
    perspective: Perspective | None = None
    fisheye: Fisheye | None = None
    # Adobe Lightroom no longer relies on camera profiles to correct aberration.


def load(text: str) -> list[Profile]:
    """Parse LCP XML text into profile descriptions.

    The input is the XML/XMP text from a ``.lcp`` file. The returned list
    contains one :class:`Profile` for each ``rdf:li/rdf:Description`` under
    ``photoshop:CameraProfiles``. Unsupported ``stCamera:*`` attributes are
    ignored with a warning.
    """

    root = ET.fromstring(text)
    return [
        _profile(element)
        for element in root.findall(".//rdf:li/rdf:Description", NS)
    ]


def dump(profile: Profile | Sequence[Profile]) -> str:
    """Serialize one profile or a sequence of profiles to LCP XML text.

    The output is an XMP/RDF document containing ``photoshop:CameraProfiles``
    with one ``rdf:li/rdf:Description`` per profile. Perspective and fisheye
    models are written as nested ``stCamera`` model elements, and vignette
    models are written inside their associated geometric model.
    """

    profiles = [profile] if isinstance(profile, Profile) else list(profile)

    root = ET.Element(
        f"{{{X_NS}}}xmpmeta",
        {f"{{{X_NS}}}xmptk": DEFAULT_XMP_TOOLKIT},
    )
    rdf = ET.SubElement(root, f"{{{RDF_NS}}}RDF")
    description = ET.SubElement(
        rdf,
        f"{{{RDF_NS}}}Description",
        {f"{{{RDF_NS}}}about": ""},
    )
    camera_profiles = ET.SubElement(
        description,
        f"{{{PHOTOSHOP_NS}}}CameraProfiles",
    )
    seq = ET.SubElement(camera_profiles, f"{{{RDF_NS}}}Seq")

    for item in profiles:
        li = ET.SubElement(seq, f"{{{RDF_NS}}}li")
        item_element = ET.SubElement(
            li,
            f"{{{RDF_NS}}}Description",
            _xml_attrs(_profile_attrs(item)),
        )
        if item.perspective is not None:
            _append_model(
                item_element,
                "PerspectiveModel",
                _perspective_attrs(item.perspective),
                item.perspective.vignette,
            )
        if item.fisheye is not None:
            _append_model(
                item_element,
                "FisheyeModel",
                _fisheye_attrs(item.fisheye),
                item.fisheye.vignette,
            )

    ET.indent(root, space=" ")
    return ET.tostring(root, encoding="unicode", short_empty_elements=True) + "\n"


def _profile(element: ET.Element) -> Profile:
    attrs = _st_attrs(element)
    known = {
        "Author",
        "Make",
        "Model",
        "UniqueCameraModel",
        "CameraPrettyName",
        "Lens",
        "LensInfo",
        "LensID",
        "LensPrettyName",
        "ImageWidth",
        "ImageLength",
        "XResolution",
        "YResolution",
        "FocalLength",
        "ApertureValue",
        "CameraRawProfile",
        "FocusDistance",
        "SensorFormatFactor",
        "PreferMetadataDistort",
        "ProfileName",
    }
    _warn_missing_fields("profile", attrs, known)
    return Profile(
        author=attrs.get("Author"),
        make=attrs["Make"],
        model=attrs.get("Model"),
        unique_camera_model=attrs.get("UniqueCameraModel"),
        camera_pretty_name=attrs["CameraPrettyName"],
        lens=attrs.get("Lens"),
        lens_info=attrs.get("LensInfo"),
        lens_id=attrs.get("LensID"),
        lens_pretty_name=attrs["LensPrettyName"],
        profile_name=attrs["ProfileName"],
        image_width=_optional(int, attrs.get("ImageWidth")),
        image_length=_optional(int, attrs.get("ImageLength")),
        x_resolution=_optional(float, attrs.get("XResolution")),
        y_resolution=_optional(float, attrs.get("YResolution")),
        focal_length=float(attrs["FocalLength"]),
        aperture_value=float(attrs["ApertureValue"]),
        camera_raw_profile=_bool(attrs["CameraRawProfile"]),
        focus_distance=float(attrs["FocusDistance"]),
        sensor_format_factor=_optional(float, attrs.get("SensorFormatFactor")),
        prefer_metadata_distort=_optional(_bool, attrs.get("PreferMetadataDistort")),
        perspective=_perspective(element.find("stCamera:PerspectiveModel", NS)),
        fisheye=_fisheye(element.find("stCamera:FisheyeModel", NS)),
    )


def _perspective(element: ET.Element | None) -> Perspective | None:
    source = _model_resource_element(element)
    if source is None:
        return None

    attrs = _st_attrs(source)
    known = {
        "Version",
        "FocalLengthX",
        "FocalLengthY",
        "ImageXCenter",
        "ImageYCenter",
        "ScaleFactor",
        "RadialDistortParam1",
        "RadialDistortParam2",
        "RadialDistortParam3",
        "TangentialDistortParam1",
        "TangentialDistortParam2",
        "ResidualMeanError",
        "ResidualStandardDeviation",
    }
    _warn_missing_fields("perspective model", attrs, known)
    return Perspective(
        version=int(attrs["Version"]),
        focal_length_x=_optional(float, attrs.get("FocalLengthX")),
        focal_length_y=_optional(float, attrs.get("FocalLengthY")),
        image_x_center=_optional(float, attrs.get("ImageXCenter")),
        image_y_center=_optional(float, attrs.get("ImageYCenter")),
        scale_factor=_optional(float, attrs.get("ScaleFactor")),
        radial_distort_param1=float(attrs["RadialDistortParam1"]),
        radial_distort_param2=_optional(
            float, attrs.get("RadialDistortParam2")
        ),
        radial_distort_param3=_optional(
            float, attrs.get("RadialDistortParam3")
        ),
        tangential_distort_param1=_optional(
            float, attrs.get("TangentialDistortParam1")
        ),
        tangential_distort_param2=_optional(
            float, attrs.get("TangentialDistortParam2")
        ),
        residual_mean_error=_optional(float, attrs.get("ResidualMeanError")),
        residual_standard_deviation=_optional(
            float, attrs.get("ResidualStandardDeviation")
        ),
        vignette=_vignette(source.find("stCamera:VignetteModel", NS)),
    )


def _fisheye(element: ET.Element | None) -> Fisheye | None:
    source = _model_resource_element(element)
    if source is None:
        return None

    attrs = _st_attrs(source)
    known = {
        "Version",
        "FocalLengthX",
        "FocalLengthY",
        "ImageXCenter",
        "ImageYCenter",
        "RadialDistortParam1",
        "RadialDistortParam2",
        "ResidualMeanError",
        "ResidualStandardDeviation",
    }
    _warn_missing_fields("fisheye model", attrs, known)
    return Fisheye(
        version=int(attrs["Version"]),
        focal_length_x=_optional(float, attrs.get("FocalLengthX")),
        focal_length_y=_optional(float, attrs.get("FocalLengthY")),
        image_x_center=_optional(float, attrs.get("ImageXCenter")),
        image_y_center=_optional(float, attrs.get("ImageYCenter")),
        radial_distort_param1=float(attrs["RadialDistortParam1"]),
        radial_distort_param2=_optional(float, attrs.get("RadialDistortParam2")),
        residual_mean_error=_optional(float, attrs.get("ResidualMeanError")),
        residual_standard_deviation=_optional(
            float, attrs.get("ResidualStandardDeviation")
        ),
        vignette=_vignette(source.find("stCamera:VignetteModel", NS)),
    )


def _vignette(element: ET.Element | None) -> Vignette | None:
    source = _model_resource_element(element)
    if source is None:
        return None

    attrs = _st_attrs(source)
    known = {
        "FocalLengthX",
        "FocalLengthY",
        "ImageXCenter",
        "ImageYCenter",
        "VignetteModelParam1",
        "VignetteModelParam2",
        "VignetteModelParam3",
        "ResidualMeanError",
    }
    _warn_missing_fields("vignette model", attrs, known)

    piecewise_params = []
    for item in source.findall(
        ".//stCamera:VignetteModelPiecewiseParam/rdf:Seq/rdf:li", NS
    ):
        if item.text is None:
            continue
        x, y = (part.strip() for part in item.text.split(",", 1))
        piecewise_params.append((float(x), float(y)))

    return Vignette(
        focal_length_x=_optional(float, attrs.get("FocalLengthX")),
        focal_length_y=_optional(float, attrs.get("FocalLengthY")),
        image_x_center=_optional(float, attrs.get("ImageXCenter")),
        image_y_center=_optional(float, attrs.get("ImageYCenter")),
        param1=float(attrs["VignetteModelParam1"]),
        param2=_optional(float, attrs.get("VignetteModelParam2")),
        param3=_optional(float, attrs.get("VignetteModelParam3")),
        residual_mean_error=_optional(float, attrs.get("ResidualMeanError")),
        piecewise_params=piecewise_params,
    )


def _bool(value: str) -> bool:
    return value.lower() == "true"


def _optional[S, T](func: Callable[[S], T], value: S | None) -> T | None:
    if value is None:
        return None
    return func(value)


def _profile_attrs(profile: Profile) -> dict[str, str]:
    attrs = {}
    _set_attr(attrs, "Author", profile.author)
    _set_attr(attrs, "Make", profile.make)
    _set_attr(attrs, "Model", profile.model)
    _set_attr(attrs, "UniqueCameraModel", profile.unique_camera_model)
    _set_attr(attrs, "CameraPrettyName", profile.camera_pretty_name)
    _set_attr(attrs, "Lens", profile.lens)
    _set_attr(attrs, "LensInfo", profile.lens_info)
    _set_attr(attrs, "LensID", profile.lens_id)
    _set_attr(attrs, "LensPrettyName", profile.lens_pretty_name)
    _set_attr(attrs, "ImageWidth", profile.image_width)
    _set_attr(attrs, "ImageLength", profile.image_length)
    _set_attr(attrs, "XResolution", profile.x_resolution)
    _set_attr(attrs, "YResolution", profile.y_resolution)
    _set_attr(attrs, "FocalLength", profile.focal_length)
    _set_attr(attrs, "ApertureValue", profile.aperture_value)
    _set_attr(attrs, "CameraRawProfile", profile.camera_raw_profile)
    _set_attr(attrs, "FocusDistance", profile.focus_distance)
    _set_attr(attrs, "SensorFormatFactor", profile.sensor_format_factor)
    _set_attr(attrs, "PreferMetadataDistort", profile.prefer_metadata_distort)
    _set_attr(attrs, "ProfileName", profile.profile_name)
    return attrs


def _perspective_attrs(perspective: Perspective) -> dict[str, str]:
    attrs = {}
    _set_attr(attrs, "Version", perspective.version)
    _set_attr(attrs, "FocalLengthX", perspective.focal_length_x)
    _set_attr(attrs, "FocalLengthY", perspective.focal_length_y)
    _set_attr(attrs, "ImageXCenter", perspective.image_x_center)
    _set_attr(attrs, "ImageYCenter", perspective.image_y_center)
    _set_attr(attrs, "ScaleFactor", perspective.scale_factor)
    _set_attr(attrs, "RadialDistortParam1", perspective.radial_distort_param1)
    _set_attr(attrs, "RadialDistortParam2", perspective.radial_distort_param2)
    _set_attr(attrs, "RadialDistortParam3", perspective.radial_distort_param3)
    _set_attr(
        attrs,
        "TangentialDistortParam1",
        perspective.tangential_distort_param1,
    )
    _set_attr(
        attrs,
        "TangentialDistortParam2",
        perspective.tangential_distort_param2,
    )
    _set_attr(attrs, "ResidualMeanError", perspective.residual_mean_error)
    _set_attr(
        attrs,
        "ResidualStandardDeviation",
        perspective.residual_standard_deviation,
    )
    return attrs


def _fisheye_attrs(fisheye: Fisheye) -> dict[str, str]:
    attrs = {}
    _set_attr(attrs, "Version", fisheye.version)
    _set_attr(attrs, "FocalLengthX", fisheye.focal_length_x)
    _set_attr(attrs, "FocalLengthY", fisheye.focal_length_y)
    _set_attr(attrs, "ImageXCenter", fisheye.image_x_center)
    _set_attr(attrs, "ImageYCenter", fisheye.image_y_center)
    _set_attr(attrs, "RadialDistortParam1", fisheye.radial_distort_param1)
    _set_attr(attrs, "RadialDistortParam2", fisheye.radial_distort_param2)
    _set_attr(attrs, "ResidualMeanError", fisheye.residual_mean_error)
    _set_attr(
        attrs,
        "ResidualStandardDeviation",
        fisheye.residual_standard_deviation,
    )
    return attrs


def _vignette_attrs(vignette: Vignette) -> dict[str, str]:
    attrs = {}
    _set_attr(attrs, "FocalLengthX", vignette.focal_length_x)
    _set_attr(attrs, "FocalLengthY", vignette.focal_length_y)
    _set_attr(attrs, "ImageXCenter", vignette.image_x_center)
    _set_attr(attrs, "ImageYCenter", vignette.image_y_center)
    _set_attr(attrs, "VignetteModelParam1", vignette.param1)
    _set_attr(attrs, "VignetteModelParam2", vignette.param2)
    _set_attr(attrs, "VignetteModelParam3", vignette.param3)
    _set_attr(attrs, "ResidualMeanError", vignette.residual_mean_error)
    return attrs


def _append_vignette(parent: ET.Element, vignette: Vignette) -> None:
    if not vignette.piecewise_params:
        ET.SubElement(
            parent,
            f"{{{ST_CAMERA_NS}}}VignetteModel",
            _xml_attrs(_vignette_attrs(vignette)),
        )
        return

    vignette_element = ET.SubElement(parent, f"{{{ST_CAMERA_NS}}}VignetteModel")
    description = ET.SubElement(
        vignette_element,
        f"{{{RDF_NS}}}Description",
        _xml_attrs(_vignette_attrs(vignette)),
    )
    piecewise = ET.SubElement(
        description,
        f"{{{ST_CAMERA_NS}}}VignetteModelPiecewiseParam",
    )
    seq = ET.SubElement(piecewise, f"{{{RDF_NS}}}Seq")
    for x, y in vignette.piecewise_params:
        item = ET.SubElement(seq, f"{{{RDF_NS}}}li")
        item.text = f"{x:.6f}, {y:.6f}"


def _append_model(
    parent: ET.Element,
    tag_name: str,
    attrs: dict[str, str],
    vignette: Vignette | None,
) -> None:
    if vignette is None:
        ET.SubElement(parent, f"{{{ST_CAMERA_NS}}}{tag_name}", _xml_attrs(attrs))
        return

    model = ET.SubElement(parent, f"{{{ST_CAMERA_NS}}}{tag_name}")
    description = ET.SubElement(model, f"{{{RDF_NS}}}Description", _xml_attrs(attrs))
    _append_vignette(description, vignette)


def _model_resource_element(element: ET.Element | None) -> ET.Element | None:
    if element is None:
        return None
    description = element.find("rdf:Description", NS)
    return description if description is not None else element


def _st_attrs(element: ET.Element | None) -> dict[str, str]:
    if element is None:
        return {}
    return {
        name.rsplit("}", 1)[-1]: value
        for name, value in element.attrib.items()
        if name.startswith(f"{{{ST_CAMERA_NS}}}")
    }


def _warn_missing_fields(context: str, attrs: dict[str, str], known: set[str]) -> None:
    missing = sorted(set(attrs) - known)
    if missing:
        warn(
            f"Unsupported {context} field(s) ignored: {', '.join(missing)}",
            stacklevel=3,
        )


def _set_attr(attrs: dict[str, str], name: str, value: object) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        attrs[name] = "True" if value else "False"
    elif isinstance(value, float):
        attrs[name] = format(value, ".15g")
    else:
        attrs[name] = str(value)


def _xml_attrs(attrs: dict[str, str]) -> dict[str, str]:
    return {f"{{{ST_CAMERA_NS}}}{name}": value for name, value in attrs.items()}
