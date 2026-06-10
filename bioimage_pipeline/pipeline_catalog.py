"""Curated CellProfiler module catalog for the Phase 15.2 GUI builder.

The catalog intentionally stores only lightweight metadata that our GUI needs
to build a conservative CellProfiler headless pipeline. It is not a vendored
copy of CellProfiler internals.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SettingVisibility:
    """Visibility rule for a module setting in the GUI."""

    mode: str = "always"
    setting_label: str | None = None
    values: tuple[str, ...] = field(default_factory=tuple)

    def is_visible(self, settings: dict[str, str]) -> bool:
        """Return whether a setting should be displayed for current values."""
        if self.mode == "always":
            return True
        if self.mode != "conditional" or self.setting_label is None:
            return False
        return settings.get(self.setting_label) in self.values


ALWAYS_VISIBLE = SettingVisibility()


def visible_when(setting_label: str, *values: str) -> SettingVisibility:
    """Create a conditional visibility rule."""
    return SettingVisibility(
        mode="conditional",
        setting_label=setting_label,
        values=tuple(values),
    )


@dataclass(frozen=True)
class ModuleParameter:
    """Display metadata for a CellProfiler module setting."""

    label: str
    default: str
    description: str = ""
    choices: tuple[str, ...] = field(default_factory=tuple)
    visibility: SettingVisibility = ALWAYS_VISIBLE
    internal: bool = False

    @property
    def name(self) -> str:
        """Backward-compatible alias for older builder code."""
        return self.label


@dataclass(frozen=True)
class ModuleDefinition:
    """A CellProfiler module exposed by the GUI builder."""

    name: str
    display_name: str
    category: str
    description: str
    parameters: tuple[ModuleParameter, ...] = field(default_factory=tuple)
    variable_revision_number: int = 1

    def visible_parameters(self, settings: dict[str, str] | None = None) -> list[ModuleParameter]:
        """Return settings visible for the provided current setting values."""
        current = {
            parameter.label: parameter.default
            for parameter in self.parameters
        }
        if settings:
            current.update(settings)
        return [
            parameter
            for parameter in self.parameters
            if not parameter.internal and parameter.visibility.is_visible(current)
        ]


_MODULES: tuple[ModuleDefinition, ...] = (
    ModuleDefinition(
        name="Images",
        display_name="Images",
        category="Input",
        description="Collect image file names for a CellProfiler pipeline.",
        parameters=(
            ModuleParameter(
                "Input folder path",
                "",
                "Folder containing input images for this pipeline (GUI-managed).",
            ),
            ModuleParameter(
                "Filter images?",
                "Images only",
                "Limit the input set before downstream modules see files.",
                ("Images only", "No filtering", "Custom"),
            ),
            ModuleParameter(
                "Select the rule criteria",
                'and (extension does isimage) (directory doesnot containregexp "[\\\\\\\\/]\\\\.")',
                "Rule expression used when image filtering is enabled.",
            ),
        ),
        variable_revision_number=2,
    ),
    ModuleDefinition(
        name="Metadata",
        display_name="Metadata",
        category="Input",
        description="Extract metadata from image file names or folders.",
        parameters=(
            ModuleParameter(
                "Extract metadata?",
                "No",
                "Enable metadata extraction from file names or folder names.",
                ("Yes", "No"),
            ),
            ModuleParameter(
                "Metadata data type",
                "Text",
                "Data type for extracted metadata values.",
                ("Text", "Integer", "Float"),
                visible_when("Extract metadata?", "Yes"),
            ),
            ModuleParameter(
                "Metadata types",
                "{}",
                "Saved CellProfiler metadata type mapping.",
                visibility=visible_when("Extract metadata?", "Yes"),
                internal=True,
            ),
            ModuleParameter(
                "Extraction method count",
                "1",
                "Number of metadata extraction methods configured.",
                visibility=visible_when("Extract metadata?", "Yes"),
                internal=True,
            ),
            ModuleParameter(
                "Metadata extraction method",
                "Extract from image file headers",
                "Source for metadata values.",
                ("Extract from image file headers", "Extract from file/folder names", "Import from file"),
                visible_when("Extract metadata?", "Yes"),
            ),
            ModuleParameter(
                "Metadata source",
                "File name",
                "Where to apply the extraction expression.",
                ("File name", "Folder name"),
                visible_when("Extract metadata?", "Yes"),
            ),
            ModuleParameter(
                "Regular expression to extract from file name",
                "^(?P<Well>[A-H][0-9]{2})",
                "Named groups become metadata columns.",
                visibility=visible_when("Extract metadata?", "Yes"),
            ),
            ModuleParameter(
                "Regular expression to extract from folder name",
                "(?P<Date>[0-9]{4}_[0-9]{2}_[0-9]{2})$",
                "Named groups become metadata columns.",
                visibility=visible_when("Extract metadata?", "Yes"),
            ),
            ModuleParameter(
                "Extract metadata from",
                "All images",
                "Image subset used for metadata extraction.",
                ("All images", "Images matching rules"),
                visible_when("Extract metadata?", "Yes"),
            ),
            ModuleParameter(
                "Select the filtering criteria",
                'and (file does contain "")',
                "Metadata extraction filter criteria.",
                visibility=visible_when("Extract metadata from", "Images matching rules"),
            ),
            ModuleParameter(
                "Metadata file location",
                "Elsewhere...|",
                "Location for external metadata file imports.",
                visibility=visible_when("Metadata extraction method", "Import from file"),
            ),
            ModuleParameter(
                "Match file and image metadata",
                "[]",
                "External metadata matching rules.",
                visibility=visible_when("Metadata extraction method", "Import from file"),
                internal=True,
            ),
            ModuleParameter(
                "Use case insensitive matching?",
                "No",
                "Match external metadata without case sensitivity.",
                ("Yes", "No"),
                visible_when("Metadata extraction method", "Import from file"),
            ),
            ModuleParameter(
                "Metadata file name",
                "",
                "External metadata file name.",
                visibility=visible_when("Metadata extraction method", "Import from file"),
            ),
            ModuleParameter(
                "Does cached metadata exist?",
                "No",
                "CellProfiler cache marker.",
                ("Yes", "No"),
                internal=True,
            ),
        ),
        variable_revision_number=6,
    ),
    ModuleDefinition(
        name="NamesAndTypes",
        display_name="NamesAndTypes",
        category="Input",
        description="Assign image names and types before analysis.",
        parameters=(
            ModuleParameter(
                "Assign a name to",
                "Images matching rules",
                "Choose which images this assignment applies to.",
                ("Images matching rules", "All images"),
            ),
            ModuleParameter(
                "Select the image type",
                "Grayscale image",
                "How CellProfiler should interpret matching images.",
                ("Grayscale image", "Color image", "Mask", "Illumination function"),
            ),
            ModuleParameter(
                "Name to assign these images",
                "DNA",
                "Image name used by analysis modules.",
            ),
            ModuleParameter(
                "Match metadata",
                "[]",
                "Metadata matching rules for image sets.",
                internal=True,
            ),
            ModuleParameter(
                "Image set matching method",
                "Order",
                "How images are assembled into image sets.",
                ("Order", "Metadata"),
            ),
            ModuleParameter(
                "Set intensity range from",
                "Image metadata",
                "Source for display/intensity scaling.",
                ("Image metadata", "Manual"),
            ),
            ModuleParameter(
                "Assignments count",
                "1",
                "Number of image assignments in this module.",
                internal=True,
            ),
            ModuleParameter(
                "Single images count",
                "0",
                "Number of single-image assignments.",
                internal=True,
            ),
            ModuleParameter(
                "Maximum intensity",
                "255.0",
                "Manual maximum intensity.",
                visibility=visible_when("Set intensity range from", "Manual"),
            ),
            ModuleParameter(
                "Process as 3D?",
                "No",
                "Treat images as 3D volumes.",
                ("Yes", "No"),
            ),
            ModuleParameter("Relative pixel spacing in X", "1.0"),
            ModuleParameter("Relative pixel spacing in Y", "1.0"),
            ModuleParameter("Relative pixel spacing in Z", "1.0"),
            ModuleParameter(
                "Select the rule criteria",
                "and (extension does isimage)",
                "Rules used to select files for this image name.",
                visibility=visible_when("Assign a name to", "Images matching rules"),
            ),
            ModuleParameter(
                "Name to assign these images",
                "DNA",
                "Per-assignment image name used by analysis modules.",
                internal=True,
            ),
            ModuleParameter(
                "Name to assign these objects",
                "Cell",
                "Object name placeholder used by object-image assignments.",
                internal=True,
            ),
            ModuleParameter(
                "Select the image type",
                "Grayscale image",
                "Per-assignment image type.",
                ("Grayscale image", "Color image", "Mask", "Illumination function"),
                internal=True,
            ),
            ModuleParameter(
                "Set intensity range from",
                "Image metadata",
                "Per-assignment intensity range source.",
                ("Image metadata", "Manual"),
                internal=True,
            ),
            ModuleParameter("Maximum intensity", "255.0", internal=True),
        ),
        variable_revision_number=8,
    ),
    ModuleDefinition(
        name="Groups",
        display_name="Groups",
        category="Input",
        description="Group images for batch processing.",
        parameters=(ModuleParameter("Do you want to group your images?", "No"),),
    ),
    ModuleDefinition(
        name="IdentifyPrimaryObjects",
        display_name="IdentifyPrimaryObjects",
        category="Object Processing",
        description="Identify primary objects such as nuclei.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the primary objects to be identified", "Nuclei"),
            ModuleParameter("Typical diameter of objects, in pixel units (Min,Max)", "10,40"),
            ModuleParameter(
                "Discard objects outside the diameter range?",
                "Yes",
                "Remove objects smaller or larger than the configured diameter.",
                ("Yes", "No"),
            ),
            ModuleParameter(
                "Discard objects touching the border of the image?",
                "Yes",
                "Remove partial edge objects.",
                ("Yes", "No"),
            ),
            ModuleParameter(
                "Method to distinguish clumped objects",
                "Shape",
                "Split touching primary objects.",
                ("Shape", "Intensity", "None"),
            ),
            ModuleParameter(
                "Method to draw dividing lines between clumped objects",
                "Shape",
                "Draw object boundaries after clump detection.",
                ("Shape", "Intensity", "None"),
            ),
            ModuleParameter("Size of smoothing filter", "10"),
            ModuleParameter(
                "Suppress local maxima that are closer than this minimum allowed distance",
                "5",
            ),
            ModuleParameter(
                "Speed up by using lower-resolution image to find local maxima?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter(
                "Fill holes in identified objects?",
                "After declumping only",
                choices=("After declumping only", "Never", "Before declumping"),
            ),
            ModuleParameter(
                "Automatically calculate size of smoothing filter for declumping?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter(
                "Automatically calculate minimum allowed distance between local maxima?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter(
                "Handling of objects if excessive number of objects identified",
                "Continue",
                choices=("Continue", "Erase", "Truncate"),
            ),
            ModuleParameter("Maximum number of objects", "500"),
            ModuleParameter(
                "Display accepted local maxima?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter("Select maxima color", "Blue"),
            ModuleParameter(
                "Use advanced settings?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter("Threshold setting version", "11", internal=True),
            ModuleParameter(
                "Threshold strategy",
                "Global",
                "Use one threshold for the image or adapt locally.",
                ("Global", "Adaptive"),
            ),
            ModuleParameter(
                "Thresholding method",
                "Otsu",
                "Automatic threshold method.",
                ("Otsu", "Minimum Cross-Entropy", "Manual"),
            ),
            ModuleParameter("Threshold smoothing scale", "1.3488"),
            ModuleParameter(
                "Threshold correction factor",
                "1",
                "Multiplier applied to the calculated threshold.",
            ),
            ModuleParameter(
                "Lower and upper bounds on threshold",
                "0,1",
                "Clamp the calculated threshold into this range.",
            ),
            ModuleParameter(
                "Manual threshold",
                "0.0",
                "Threshold value used with manual thresholding.",
                visibility=visible_when("Thresholding method", "Manual"),
            ),
            ModuleParameter(
                "Select the measurement to threshold with",
                "None",
                visibility=visible_when("Thresholding method", "Measurement"),
            ),
            ModuleParameter(
                "Two-class or three-class thresholding?",
                "Two classes",
                choices=("Two classes", "Three classes"),
            ),
            ModuleParameter(
                "Assign pixels in the middle intensity class to the foreground or the background?",
                "Foreground",
                choices=("Foreground", "Background"),
            ),
            ModuleParameter("Size of adaptive window", "10"),
            ModuleParameter("Lower outlier fraction", "0.05"),
            ModuleParameter("Upper outlier fraction", "0.05"),
            ModuleParameter("Averaging method", "Mean", choices=("Mean", "Median", "Mode")),
            ModuleParameter(
                "Variance method",
                "Standard deviation",
                choices=("Standard deviation", "Median absolute deviation"),
            ),
            ModuleParameter("# of deviations", "2"),
        ),
        variable_revision_number=14,
    ),
    ModuleDefinition(
        name="IdentifySecondaryObjects",
        display_name="IdentifySecondaryObjects",
        category="Object Processing",
        description="Identify secondary objects around primary objects.",
        parameters=(
            ModuleParameter("Select the input objects", "Nuclei"),
            ModuleParameter("Name the objects to be identified", "Cells"),
            ModuleParameter("Select the method to identify the secondary objects", "Propagation"),
        ),
    ),
    ModuleDefinition(
        name="MeasureObjectSizeShape",
        display_name="MeasureObjectSizeShape",
        category="Measurement",
        description="Measure object size and shape features.",
        parameters=(ModuleParameter("Select objects to measure", "Nuclei"),),
    ),
    ModuleDefinition(
        name="MeasureObjectIntensity",
        display_name="MeasureObjectIntensity",
        category="Measurement",
        description="Measure intensity features for objects.",
        parameters=(
            ModuleParameter("Select images to measure", "DNA"),
            ModuleParameter("Select objects to measure", "Nuclei"),
        ),
    ),
    ModuleDefinition(
        name="MeasureTexture",
        display_name="MeasureTexture",
        category="Measurement",
        description="Measure texture features for images or objects.",
        parameters=(
            ModuleParameter("Select images to measure", "DNA"),
            ModuleParameter("Texture scale to measure", "3"),
        ),
    ),
    ModuleDefinition(
        name="RelateObjects",
        display_name="RelateObjects",
        category="Object Processing",
        description="Relate child objects to parent objects.",
        parameters=(
            ModuleParameter("Select the input child objects", "Nuclei"),
            ModuleParameter("Select the input parent objects", "Cells"),
        ),
    ),
    ModuleDefinition(
        name="ClassifyObjects",
        display_name="ClassifyObjects",
        category="Object Processing",
        description="Classify objects using CellProfiler measurements.",
        parameters=(ModuleParameter("Select the object to be classified", "Nuclei"),),
    ),
    ModuleDefinition(
        name="SaveImages",
        display_name="SaveImages",
        category="File Processing",
        description="Save image, mask, or object outputs as files.",
        parameters=(
            ModuleParameter(
                "Select the type of image to save",
                "Image",
                "Choose whether to save images, masks, or labeled objects.",
                ("Image", "Mask", "Objects"),
            ),
            ModuleParameter(
                "Select the image to save",
                "DNA",
                "Image or mask name to save.",
                visibility=visible_when("Select the type of image to save", "Image", "Mask"),
            ),
            ModuleParameter(
                "Select method for constructing file names",
                "Sequential numbers",
                "How output file names are created.",
                ("Sequential numbers", "Single name", "From image filename"),
            ),
            ModuleParameter("Select image name for file prefix", "DNA"),
            ModuleParameter("Enter file prefix", "Nuclei"),
            ModuleParameter("Number of digits", "4"),
            ModuleParameter(
                "Append a suffix to the image file name?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter("Text to append to the image name", "Objects"),
            ModuleParameter(
                "Saved file format",
                "tiff",
                "Output image file format.",
                ("tiff", "png", "jpeg"),
            ),
            ModuleParameter(
                "Output file location",
                "Default Output Folder|None",
                "Folder CellProfiler writes to during headless execution.",
                ("Default Output Folder|None", "Default Input Folder|None"),
            ),
            ModuleParameter("Image bit depth", "16-bit integer"),
            ModuleParameter(
                "Overwrite existing files without warning?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter("When to save", "Every cycle"),
            ModuleParameter(
                "Record the file and path information to the saved image?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter(
                "Create subfolders in the output folder?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter("Base image folder", "Default Input Folder"),
            ModuleParameter("How to save the series", "T (Time)"),
            ModuleParameter(
                "Enter single file name",
                "Nuclei",
                "Base file name for single-name output.",
                visibility=visible_when("Select method for constructing file names", "Single name"),
            ),
        ),
        variable_revision_number=15,
    ),
    ModuleDefinition(
        name="ExportToSpreadsheet",
        display_name="ExportToSpreadsheet",
        category="File Processing",
        description="Export measurements to CSV files.",
        parameters=(
            ModuleParameter(
                "Select the column delimiter",
                'Comma (",")',
                "Delimiter used in exported measurement files.",
                ('Comma (",")', "Tab"),
            ),
            ModuleParameter(
                "Add image metadata columns to your object data file?",
                "No",
                "Include image metadata columns in object tables.",
                ("Yes", "No"),
            ),
            ModuleParameter(
                "Add image file and folder names to your object data file?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter(
                "Select the measurements to export",
                "No",
                "Whether to export all measurements or a selected subset.",
                ("Yes", "No"),
            ),
            ModuleParameter(
                "Calculate the per-image mean values for object measurements?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter(
                "Calculate the per-image median values for object measurements?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter(
                "Calculate the per-image standard deviation values for object measurements?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter(
                "Output file location",
                "Default Output Folder|.",
                "Folder for exported CSV files.",
                ("Default Output Folder|.", "Default Input Folder|."),
            ),
            ModuleParameter("Create a GenePattern GCT file?", "No", choices=("Yes", "No")),
            ModuleParameter("Select source of sample row name", "Metadata"),
            ModuleParameter("Select the image to use as the identifier", "None"),
            ModuleParameter("Select the metadata to use as the identifier", "None"),
            ModuleParameter("Export all measurement types?", "No", choices=("Yes", "No")),
            ModuleParameter("Press button to select measurements", "None|None", internal=True),
            ModuleParameter("Representation of Nan/Inf", "NaN"),
            ModuleParameter("Add a prefix to file names?", "No", choices=("Yes", "No")),
            ModuleParameter("Filename prefix", "MyExpt_"),
            ModuleParameter(
                "Overwrite existing files without warning?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter("Data to export", "Image"),
            ModuleParameter(
                "Combine these object measurements with those of the previous object?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter("File name", "Image.csv"),
            ModuleParameter(
                "Use the object name for the file name?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter("Data to export", "Nuclei"),
            ModuleParameter(
                "Combine these object measurements with those of the previous object?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter("File name", "Nuclei.csv"),
            ModuleParameter(
                "Use the object name for the file name?",
                "No",
                choices=("Yes", "No"),
            ),
        ),
        variable_revision_number=13,
    ),
    # ---- Image Processing -------------------------------------------------
    ModuleDefinition(
        name="ColorToGray",
        display_name="ColorToGray",
        category="Image Processing",
        description="Convert a color image to one or more grayscale images.",
        parameters=(
            ModuleParameter("Select the input image", "OrigColor"),
            ModuleParameter(
                "Conversion method",
                "Combine",
                "Combine channels into one image or split into separate images.",
                ("Combine", "Split"),
            ),
            ModuleParameter(
                "Image type",
                "RGB",
                choices=("RGB", "HSV", "Channels"),
            ),
            ModuleParameter("Name the output image", "OrigGray"),
        ),
    ),
    ModuleDefinition(
        name="GrayToColor",
        display_name="GrayToColor",
        category="Image Processing",
        description="Combine grayscale images into a single color image.",
        parameters=(
            ModuleParameter(
                "Select a color scheme",
                "RGB",
                choices=("RGB", "CMYK", "Composite"),
            ),
            ModuleParameter("Name the output image", "ColorImage"),
        ),
    ),
    ModuleDefinition(
        name="InvertForPrinting",
        display_name="InvertForPrinting",
        category="Image Processing",
        description="Invert fluorescent images into a printer-friendly form.",
        parameters=(
            ModuleParameter(
                "Input image type",
                "Color",
                choices=("Color", "Grayscale"),
            ),
        ),
    ),
    ModuleDefinition(
        name="UnmixColors",
        display_name="UnmixColors",
        category="Image Processing",
        description="Separate stains in a brightfield image by color unmixing.",
        parameters=(
            ModuleParameter("Select the input color image", "OrigColor"),
            ModuleParameter("Stain count", "2"),
        ),
    ),
    ModuleDefinition(
        name="Crop",
        display_name="Crop",
        category="Image Processing",
        description="Crop an image to a rectangle, ellipse, or another shape.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "CropDNA"),
            ModuleParameter(
                "Select the cropping shape",
                "Rectangle",
                choices=("Rectangle", "Ellipse", "Image", "Objects", "Previous cropping"),
            ),
            ModuleParameter(
                "Select the cropping method",
                "Coordinates",
                choices=("Coordinates", "Mouse"),
            ),
        ),
    ),
    ModuleDefinition(
        name="Resize",
        display_name="Resize",
        category="Image Processing",
        description="Resize an image by a scale factor or to specific dimensions.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "ResizedDNA"),
            ModuleParameter(
                "Resizing method",
                "Resize by a fraction or multiple of the original size",
                choices=(
                    "Resize by a fraction or multiple of the original size",
                    "Resize by specifying desired final dimensions",
                ),
            ),
            ModuleParameter("Resizing factor", "0.25"),
        ),
    ),
    ModuleDefinition(
        name="Tile",
        display_name="Tile",
        category="Image Processing",
        description="Tile multiple images or cycles into a single large image.",
        parameters=(
            ModuleParameter("Select an input image", "DNA"),
            ModuleParameter("Name the output image", "TiledImage"),
            ModuleParameter(
                "Tile assembly method",
                "Within cycles",
                choices=("Within cycles", "Across cycles"),
            ),
        ),
    ),
    ModuleDefinition(
        name="MakeProjection",
        display_name="MakeProjection",
        category="Image Processing",
        description="Combine several images into a single projection.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter(
                "Type of projection",
                "Average",
                choices=("Average", "Maximum", "Minimum", "Sum", "Variance", "Power", "Brightfield"),
            ),
            ModuleParameter("Name the output image", "ProjectionDNA"),
        ),
    ),
    ModuleDefinition(
        name="FlipAndRotate",
        display_name="FlipAndRotate",
        category="Image Processing",
        description="Flip or rotate an image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "FlippedOrigBlue"),
            ModuleParameter(
                "Select method to flip image",
                "Do not flip",
                choices=("Do not flip", "Left to right", "Top to bottom", "Left to right and top to bottom"),
            ),
            ModuleParameter(
                "Select method to rotate image",
                "Do not rotate",
                choices=("Do not rotate", "Coordinates", "Mouse", "Angle"),
            ),
        ),
    ),
    ModuleDefinition(
        name="Smooth",
        display_name="Smooth",
        category="Image Processing",
        description="Smooth (blur) an image to reduce noise.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "FilteredImage"),
            ModuleParameter(
                "Select smoothing method",
                "Gaussian Filter",
                choices=(
                    "Fit Polynomial",
                    "Gaussian Filter",
                    "Median Filter",
                    "Smooth Keeping Edges",
                    "Circular Average Filter",
                    "Smooth to Average",
                ),
            ),
            ModuleParameter(
                "Calculate artifact diameter automatically?",
                "Yes",
                choices=("Yes", "No"),
            ),
        ),
    ),
    ModuleDefinition(
        name="GaussianFilter",
        display_name="GaussianFilter",
        category="Advanced",
        description="Blur an image with a Gaussian filter of a given sigma.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "FilteredImage"),
            ModuleParameter("Sigma", "1"),
        ),
    ),
    ModuleDefinition(
        name="MedianFilter",
        display_name="MedianFilter",
        category="Advanced",
        description="Reduce salt-and-pepper noise with a median filter.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "FilteredImage"),
            ModuleParameter("Window", "3"),
        ),
    ),
    ModuleDefinition(
        name="ReduceNoise",
        display_name="ReduceNoise",
        category="Advanced",
        description="Reduce noise using non-local means filtering.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "FilteredImage"),
            ModuleParameter("Size", "7"),
            ModuleParameter("Distance", "11"),
            ModuleParameter("Cut-off distance", "0.1"),
        ),
    ),
    ModuleDefinition(
        name="MatchTemplate",
        display_name="MatchTemplate",
        category="Advanced",
        description="Find a template within an image via cross-correlation.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "MatchedImage"),
        ),
    ),
    ModuleDefinition(
        name="EnhanceEdges",
        display_name="EnhanceEdges",
        category="Image Processing",
        description="Enhance or identify edges in an image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "EdgedImage"),
            ModuleParameter(
                "Select an edge-finding method",
                "Sobel",
                choices=("Sobel", "Prewitt", "Roberts", "LoG", "Canny", "Kirsch"),
            ),
        ),
    ),
    ModuleDefinition(
        name="EnhanceOrSuppressFeatures",
        display_name="EnhanceOrSuppressFeatures",
        category="Image Processing",
        description="Enhance or suppress speckles, neurites, or other features.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "FilteredImage"),
            ModuleParameter(
                "Select the operation",
                "Enhance",
                choices=("Enhance", "Suppress"),
            ),
            ModuleParameter(
                "Feature type",
                "Speckles",
                choices=("Speckles", "Neurites", "Dark holes", "Circles", "Texture", "DIC"),
            ),
        ),
    ),
    ModuleDefinition(
        name="ImageMath",
        display_name="ImageMath",
        category="Image Processing",
        description="Perform arithmetic between images or with constants.",
        parameters=(
            ModuleParameter(
                "Operation",
                "Add",
                choices=(
                    "Add", "Subtract", "Absolute Difference", "Multiply", "Divide",
                    "Average", "Minimum", "Maximum", "Invert", "Log transform (base 2)",
                    "And", "Or", "Not", "Equals",
                ),
            ),
            ModuleParameter("Name the output image", "ImageAfterMath"),
        ),
    ),
    ModuleDefinition(
        name="RescaleIntensity",
        display_name="RescaleIntensity",
        category="Image Processing",
        description="Rescale image intensity values into a chosen range.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "RescaledImage"),
            ModuleParameter(
                "Rescaling method",
                "Stretch each image to use the full intensity range",
                choices=(
                    "Stretch each image to use the full intensity range",
                    "Choose specific values to be reset to the full intensity range",
                    "Choose specific values to be reset to a custom range",
                    "Divide by the image's maximum",
                    "Divide each image by the same value",
                    "Divide each image by a previously calculated value",
                    "Match the image's maximum to another image's maximum",
                ),
            ),
        ),
    ),
    ModuleDefinition(
        name="MaskImage",
        display_name="MaskImage",
        category="Image Processing",
        description="Mask an image using objects or a binary image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "MaskedImage"),
            ModuleParameter(
                "Use objects or an image as a mask?",
                "Objects",
                choices=("Objects", "Image"),
            ),
            ModuleParameter("Select object for mask", "Nuclei"),
        ),
    ),
    ModuleDefinition(
        name="OverlayOutlines",
        display_name="OverlayOutlines",
        category="Image Processing",
        description="Overlay object outlines onto an image.",
        parameters=(
            ModuleParameter(
                "Display outlines on a blank image?",
                "No",
                choices=("Yes", "No"),
            ),
            ModuleParameter("Select image on which to display outlines", "DNA"),
            ModuleParameter("Name the output image", "OrigOverlay"),
            ModuleParameter(
                "Outline display mode",
                "Color",
                choices=("Color", "Grayscale"),
            ),
        ),
    ),
    ModuleDefinition(
        name="OverlayObjects",
        display_name="OverlayObjects",
        category="Image Processing",
        description="Overlay filled objects onto an image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "ObjectsOverlay"),
            ModuleParameter("Objects", "Nuclei"),
        ),
    ),
    ModuleDefinition(
        name="Morph",
        display_name="Morph",
        category="Image Processing",
        description="Apply a sequence of morphological operations to an image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "MorphImage"),
        ),
    ),
    ModuleDefinition(
        name="CorrectIlluminationCalculate",
        display_name="CorrectIlluminationCalculate",
        category="Image Processing",
        description="Calculate an illumination correction function.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "IllumBlue"),
            ModuleParameter(
                "Select how the illumination function is calculated",
                "Regular",
                choices=("Regular", "Background"),
            ),
            ModuleParameter(
                "Dilate objects in the final averaged image?",
                "No",
                choices=("Yes", "No"),
            ),
        ),
    ),
    ModuleDefinition(
        name="CorrectIlluminationApply",
        display_name="CorrectIlluminationApply",
        category="Image Processing",
        description="Apply a previously calculated illumination correction.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "CorrBlue"),
            ModuleParameter("Select the illumination function", "IllumBlue"),
            ModuleParameter(
                "Select how the illumination function is applied",
                "Divide",
                choices=("Divide", "Subtract"),
            ),
        ),
    ),
    ModuleDefinition(
        name="Threshold",
        display_name="Threshold",
        category="Image Processing",
        description="Apply a threshold to produce a binary image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "ThreshBlue"),
            ModuleParameter(
                "Threshold strategy",
                "Global",
                choices=("Global", "Adaptive"),
            ),
            ModuleParameter(
                "Thresholding method",
                "Minimum Cross-Entropy",
                choices=("Minimum Cross-Entropy", "Otsu", "Robust Background", "Measurement", "Manual"),
            ),
            ModuleParameter("Threshold smoothing scale", "0"),
            ModuleParameter("Threshold correction factor", "1"),
            ModuleParameter("Lower and upper bounds on threshold", "0.0,1.0"),
        ),
    ),
    # ---- Object Processing ------------------------------------------------
    ModuleDefinition(
        name="IdentifyTertiaryObjects",
        display_name="IdentifyTertiaryObjects",
        category="Object Processing",
        description="Identify tertiary objects (e.g. cytoplasm) between two object sets.",
        parameters=(
            ModuleParameter("Select the larger identified objects", "Cells"),
            ModuleParameter("Select the smaller identified objects", "Nuclei"),
            ModuleParameter("Name the tertiary objects to be identified", "Cytoplasm"),
            ModuleParameter(
                "Shrink smaller object prior to subtraction?",
                "Yes",
                choices=("Yes", "No"),
            ),
        ),
    ),
    ModuleDefinition(
        name="IdentifyObjectsManually",
        display_name="IdentifyObjectsManually",
        category="Object Processing",
        description="Manually outline objects on an image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the objects to be identified", "Cells"),
        ),
    ),
    ModuleDefinition(
        name="IdentifyObjectsInGrid",
        display_name="IdentifyObjectsInGrid",
        category="Object Processing",
        description="Identify objects within a previously defined grid.",
        parameters=(
            ModuleParameter("Select the defined grid", "Grid"),
            ModuleParameter("Name the objects to be identified", "Wells"),
        ),
    ),
    ModuleDefinition(
        name="ConvertImageToObjects",
        display_name="ConvertImageToObjects",
        category="Object Processing",
        description="Convert a binary or labeled image into objects.",
        parameters=(
            ModuleParameter("Select the input image", "Binary"),
            ModuleParameter("Name the output object", "Objects"),
            ModuleParameter(
                "Convert to boolean image",
                "Yes",
                choices=("Yes", "No"),
            ),
        ),
    ),
    ModuleDefinition(
        name="ConvertObjectsToImage",
        display_name="ConvertObjectsToImage",
        category="Object Processing",
        description="Convert objects into an image (binary, grayscale, or color).",
        parameters=(
            ModuleParameter("Select the input objects", "Nuclei"),
            ModuleParameter("Name the output image", "CellImage"),
            ModuleParameter(
                "Select the color format",
                "Color",
                choices=("Color", "Binary (black & white)", "Grayscale", "uint16"),
            ),
        ),
    ),
    ModuleDefinition(
        name="CombineObjects",
        display_name="CombineObjects",
        category="Object Processing",
        description="Combine two object sets into a single set.",
        parameters=(
            ModuleParameter("Select initial object set", "Nuclei"),
            ModuleParameter("Select object set to combine", "Cells"),
            ModuleParameter("Name the combined object set", "CombinedObjects"),
        ),
    ),
    ModuleDefinition(
        name="ExpandOrShrinkObjects",
        display_name="ExpandOrShrinkObjects",
        category="Object Processing",
        description="Expand or shrink objects by a number of pixels.",
        parameters=(
            ModuleParameter("Select the input objects", "Nuclei"),
            ModuleParameter("Name the output objects", "ShrunkenNuclei"),
            ModuleParameter(
                "Select the operation",
                "Shrink objects to a point",
                choices=(
                    "Shrink objects to a point",
                    "Expand objects until touching",
                    "Add partial dividing lines between objects",
                    "Shrink objects by a specified number of pixels",
                    "Expand objects by a specified number of pixels",
                    "Skeletonize each object",
                    "Remove spurs",
                ),
            ),
        ),
    ),
    ModuleDefinition(
        name="EditObjectsManually",
        display_name="EditObjectsManually",
        category="Object Processing",
        description="Manually remove, split, or merge identified objects.",
        parameters=(
            ModuleParameter("Select the objects to be edited", "Nuclei"),
            ModuleParameter("Name the edited objects", "EditedNuclei"),
        ),
    ),
    ModuleDefinition(
        name="FilterObjects",
        display_name="FilterObjects",
        category="Object Processing",
        description="Filter objects by measurements or rules.",
        parameters=(
            ModuleParameter("Select the objects to filter", "Nuclei"),
            ModuleParameter("Name the output objects", "FilteredNuclei"),
            ModuleParameter(
                "Select the filtering method",
                "Measurements",
                choices=("Measurements", "Image or mask", "Rules", "Classifier"),
            ),
        ),
    ),
    ModuleDefinition(
        name="MaskObjects",
        display_name="MaskObjects",
        category="Object Processing",
        description="Mask objects using a region defined by another object or image.",
        parameters=(
            ModuleParameter("Select objects to be masked", "Nuclei"),
            ModuleParameter("Name the masked objects", "MaskedNuclei"),
            ModuleParameter(
                "Mask using a region defined by other objects or by binary image?",
                "Objects",
                choices=("Objects", "Image"),
            ),
        ),
    ),
    ModuleDefinition(
        name="SplitOrMergeObjects",
        display_name="SplitOrMergeObjects",
        category="Object Processing",
        description="Split or merge objects based on distance or measurements.",
        parameters=(
            ModuleParameter("Select the input objects", "Nuclei"),
            ModuleParameter("Name the new objects", "RelabeledNuclei"),
            ModuleParameter(
                "Operation",
                "Merge",
                choices=("Merge", "Split"),
            ),
        ),
    ),
    ModuleDefinition(
        name="ResizeObjects",
        display_name="ResizeObjects",
        category="Object Processing",
        description="Resize objects to a scale factor or specific dimensions.",
        parameters=(
            ModuleParameter("Select the input object", "Nuclei"),
            ModuleParameter("Name the output object", "ResizedNuclei"),
            ModuleParameter(
                "Method",
                "Factor",
                choices=("Dimensions", "Factor"),
            ),
            ModuleParameter("Factor", "0.25"),
        ),
    ),
    ModuleDefinition(
        name="TrackObjects",
        display_name="TrackObjects",
        category="Object Processing",
        description="Track objects across sequential image frames.",
        parameters=(
            ModuleParameter(
                "Choose a tracking method",
                "Overlap",
                choices=("Overlap", "Distance", "Measurements", "LAP", "Follow Neighbors"),
            ),
            ModuleParameter("Select the objects to track", "Nuclei"),
            ModuleParameter("Maximum pixel distance to consider matches", "50"),
        ),
    ),
    ModuleDefinition(
        name="Watershed",
        display_name="Watershed",
        category="Advanced",
        description="Separate touching objects using the watershed algorithm.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output object", "Nuclei"),
            ModuleParameter(
                "Generate from",
                "Distance",
                choices=("Distance", "Markers"),
            ),
        ),
    ),
    ModuleDefinition(
        name="ShrinkToObjectCenters",
        display_name="ShrinkToObjectCenters",
        category="Advanced",
        description="Shrink each object to a single point at its center.",
        parameters=(
            ModuleParameter("Select the input object", "Nuclei"),
            ModuleParameter("Name the output object", "ShrunkenNuclei"),
        ),
    ),
    ModuleDefinition(
        name="FillObjects",
        display_name="FillObjects",
        category="Advanced",
        description="Fill holes within objects below a given area.",
        parameters=(
            ModuleParameter("Select the input object", "Nuclei"),
            ModuleParameter("Name the output object", "FilledNuclei"),
            ModuleParameter("Minimum hole size", "64"),
        ),
    ),
    ModuleDefinition(
        name="ErodeObjects",
        display_name="ErodeObjects",
        category="Advanced",
        description="Shrink objects with morphological erosion.",
        parameters=(
            ModuleParameter("Select the input object", "Nuclei"),
            ModuleParameter("Name the output object", "ErodedNuclei"),
            ModuleParameter("Structuring element", "disk,1"),
        ),
    ),
    ModuleDefinition(
        name="DilateObjects",
        display_name="DilateObjects",
        category="Advanced",
        description="Grow objects with morphological dilation.",
        parameters=(
            ModuleParameter("Select the input object", "Nuclei"),
            ModuleParameter("Name the output object", "DilatedNuclei"),
            ModuleParameter("Structuring element", "disk,1"),
        ),
    ),
    # ---- Advanced image filters ------------------------------------------
    ModuleDefinition(
        name="ErodeImage",
        display_name="ErodeImage",
        category="Advanced",
        description="Erode a binary or grayscale image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "ErodedImage"),
            ModuleParameter("Structuring element", "disk,1"),
        ),
    ),
    ModuleDefinition(
        name="DilateImage",
        display_name="DilateImage",
        category="Advanced",
        description="Dilate a binary or grayscale image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "DilatedImage"),
            ModuleParameter("Structuring element", "disk,1"),
        ),
    ),
    ModuleDefinition(
        name="Opening",
        display_name="Opening",
        category="Advanced",
        description="Morphological opening (erosion followed by dilation).",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "OpenedImage"),
            ModuleParameter("Structuring element", "disk,1"),
        ),
    ),
    ModuleDefinition(
        name="Closing",
        display_name="Closing",
        category="Advanced",
        description="Morphological closing (dilation followed by erosion).",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "ClosedImage"),
            ModuleParameter("Structuring element", "disk,1"),
        ),
    ),
    ModuleDefinition(
        name="MorphologicalSkeleton",
        display_name="MorphologicalSkeleton",
        category="Advanced",
        description="Reduce shapes in a binary image to a skeleton.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "SkeletonImage"),
        ),
    ),
    ModuleDefinition(
        name="MedialAxis",
        display_name="MedialAxis",
        category="Advanced",
        description="Compute the medial axis (skeleton) of a binary image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "MedialAxisImage"),
        ),
    ),
    ModuleDefinition(
        name="RemoveHoles",
        display_name="RemoveHoles",
        category="Advanced",
        description="Fill holes in a binary image below a given size.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "FilledImage"),
            ModuleParameter("Size of holes to fill", "64"),
        ),
    ),
    ModuleDefinition(
        name="FindMaxima",
        display_name="FindMaxima",
        category="Advanced",
        description="Find local intensity maxima in an image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the output image", "Maxima"),
            ModuleParameter("Minimum distance between maxima", "5"),
        ),
    ),
    ModuleDefinition(
        name="RunImageJMacro",
        display_name="RunImageJMacro",
        category="Advanced",
        description="Run an ImageJ macro on images and retrieve the results.",
        parameters=(
            ModuleParameter("Executable directory", "Default Input Folder|"),
            ModuleParameter("ImageJ executable file", "ImageJ-win64.exe"),
        ),
    ),
    # ---- Measurement ------------------------------------------------------
    ModuleDefinition(
        name="MeasureObjectIntensityDistribution",
        display_name="MeasureObjectIntensityDistribution",
        category="Measurement",
        description="Measure the radial distribution of intensity within objects.",
        parameters=(
            ModuleParameter("Select images to measure", "DNA"),
            ModuleParameter("Hidden", "1", internal=True),
            ModuleParameter("Hidden", "1", internal=True),
            ModuleParameter("Hidden", "0", internal=True),
            ModuleParameter("Calculate intensity Zernikes?", "None"),
        ),
    ),
    ModuleDefinition(
        name="MeasureObjectNeighbors",
        display_name="MeasureObjectNeighbors",
        category="Measurement",
        description="Measure the number and adjacency of object neighbors.",
        parameters=(
            ModuleParameter("Select objects to measure", "Nuclei"),
            ModuleParameter("Select neighboring objects to measure", "Nuclei"),
            ModuleParameter(
                "Method to determine neighbors",
                "Expand until adjacent",
                choices=("Adjacent", "Expand until adjacent", "Within a specified distance"),
            ),
        ),
    ),
    ModuleDefinition(
        name="MeasureObjectOverlap",
        display_name="MeasureObjectOverlap",
        category="Measurement",
        description="Measure the overlap between two sets of objects.",
        parameters=(
            ModuleParameter("Select the objects to measure", "Nuclei"),
            ModuleParameter("Select the objects to compare", "Cells"),
        ),
    ),
    ModuleDefinition(
        name="MeasureObjectSkeleton",
        display_name="MeasureObjectSkeleton",
        category="Measurement",
        description="Measure branch and trunk features of skeletonized objects.",
        parameters=(
            ModuleParameter("Select the seed objects", "Nuclei"),
            ModuleParameter("Select the skeletonized image", "SkeletonImage"),
        ),
    ),
    ModuleDefinition(
        name="MeasureImageAreaOccupied",
        display_name="MeasureImageAreaOccupied",
        category="Measurement",
        description="Measure the total area occupied by objects or a binary image.",
        parameters=(
            ModuleParameter("Hidden", "1", internal=True),
            ModuleParameter(
                "Measure the area occupied by",
                "Objects",
                choices=("Binary Image", "Objects", "Both"),
            ),
            ModuleParameter("Select objects to measure", "Nuclei"),
        ),
    ),
    ModuleDefinition(
        name="MeasureImageIntensity",
        display_name="MeasureImageIntensity",
        category="Measurement",
        description="Measure intensity statistics across an entire image.",
        parameters=(
            ModuleParameter("Select images to measure", "DNA"),
            ModuleParameter(
                "Measure the intensity only from areas enclosed by objects?",
                "No",
                choices=("Yes", "No"),
            ),
        ),
    ),
    ModuleDefinition(
        name="MeasureImageQuality",
        display_name="MeasureImageQuality",
        category="Measurement",
        description="Measure blur, saturation, and other image quality metrics.",
        parameters=(
            ModuleParameter(
                "Calculate metrics for which images?",
                "All loaded images",
                choices=("All loaded images", "Select..."),
            ),
        ),
    ),
    ModuleDefinition(
        name="MeasureImageOverlap",
        display_name="MeasureImageOverlap",
        category="Measurement",
        description="Measure the overlap between two binary images.",
        parameters=(
            ModuleParameter("Compare segmented objects, or foreground/background?", "Foreground/background segmentation"),
            ModuleParameter("Select the image to be used as the ground truth basis", "Ground"),
            ModuleParameter("Select the image to be used to test for overlap", "Test"),
        ),
    ),
    ModuleDefinition(
        name="MeasureImageSkeleton",
        display_name="MeasureImageSkeleton",
        category="Measurement",
        description="Measure branch and trunk features of a skeleton image.",
        parameters=(
            ModuleParameter("Select an image to measure", "SkeletonImage"),
        ),
    ),
    ModuleDefinition(
        name="MeasureColocalization",
        display_name="MeasureColocalization",
        category="Measurement",
        description="Measure correlation/colocalization between image channels.",
        parameters=(
            ModuleParameter("Hidden", "2", internal=True),
            ModuleParameter("Select images to measure", "DNA"),
            ModuleParameter(
                "Set threshold as percentage of maximum intensity for the images",
                "15.0",
            ),
        ),
    ),
    ModuleDefinition(
        name="MeasureGranularity",
        display_name="MeasureGranularity",
        category="Measurement",
        description="Measure the granularity spectrum of an image.",
        parameters=(
            ModuleParameter("Select images to measure", "DNA"),
            ModuleParameter("Measure within objects?", "No", choices=("Yes", "No")),
        ),
    ),
    # ---- File Processing --------------------------------------------------
    ModuleDefinition(
        name="SaveCroppedObjects",
        display_name="SaveCroppedObjects",
        category="File Processing",
        description="Save each object as a cropped image or mask.",
        parameters=(
            ModuleParameter(
                "Do you want to save cropped images or object masks?",
                "Images",
                choices=("Images", "Masks"),
            ),
            ModuleParameter("Objects", "Nuclei"),
            ModuleParameter("Directory", "Default Output Folder|"),
        ),
    ),
    ModuleDefinition(
        name="LabelImages",
        display_name="LabelImages",
        category="File Processing",
        description="Assign plate, well, row, and column metadata to image cycles.",
        parameters=(
            ModuleParameter("Number of image sets to be processed per row", "12"),
            ModuleParameter("Order of image data", "Row"),
        ),
    ),
    ModuleDefinition(
        name="CreateBatchFiles",
        display_name="CreateBatchFiles",
        category="File Processing",
        description="Produce a batch file to run the pipeline on a cluster.",
        parameters=(
            ModuleParameter(
                "Store batch files in default output folder?",
                "Yes",
                choices=("Yes", "No"),
            ),
            ModuleParameter("Output folder path", "Default Output Folder|"),
        ),
    ),
    ModuleDefinition(
        name="ExportToDatabase",
        display_name="ExportToDatabase",
        category="File Processing",
        description="Export measurements to a MySQL or SQLite database.",
        parameters=(
            ModuleParameter(
                "Database type",
                "SQLite",
                choices=("MySQL", "SQLite", "MySQL / CSV"),
            ),
            ModuleParameter("Database name", "DefaultDB"),
        ),
    ),
    # ---- Data Tools -------------------------------------------------------
    ModuleDefinition(
        name="CalculateMath",
        display_name="CalculateMath",
        category="Data Tools",
        description="Compute new measurements from existing measurements.",
        parameters=(
            ModuleParameter("Name the output measurement", "Math"),
            ModuleParameter(
                "Operation",
                "Add",
                choices=("Add", "Subtract", "Multiply", "Divide", "None"),
            ),
        ),
    ),
    ModuleDefinition(
        name="CalculateStatistics",
        display_name="CalculateStatistics",
        category="Data Tools",
        description="Calculate Z-factor and V-factor dose-response statistics.",
        parameters=(
            ModuleParameter("Select the image measurement describing the treatment dose", "None"),
        ),
    ),
    ModuleDefinition(
        name="FlagImage",
        display_name="FlagImage",
        category="Data Tools",
        description="Flag images based on measurement criteria for QC.",
        parameters=(
            ModuleParameter("Hidden", "1", internal=True),
            ModuleParameter("Hidden", "1", internal=True),
            ModuleParameter("Name the flag's category", "Metadata"),
            ModuleParameter("Name the flag", "QCFlag"),
        ),
    ),
    ModuleDefinition(
        name="DisplayDataOnImage",
        display_name="DisplayDataOnImage",
        category="Data Tools",
        description="Overlay measurement values onto an image.",
        parameters=(
            ModuleParameter(
                "Display object or image measurements?",
                "Object",
                choices=("Image", "Object"),
            ),
            ModuleParameter("Select the input objects", "Nuclei"),
            ModuleParameter("Measurement to display", "None"),
            ModuleParameter("Select the image on which to display the measurements", "DNA"),
            ModuleParameter("Name the output image that has the measurements displayed", "DisplayImage"),
        ),
    ),
    ModuleDefinition(
        name="DisplayHistogram",
        display_name="DisplayHistogram",
        category="Data Tools",
        description="Plot a histogram of a measurement.",
        parameters=(
            ModuleParameter("Select the object whose measurements will be displayed", "Nuclei"),
            ModuleParameter("Select the measurement to display", "None"),
        ),
    ),
    ModuleDefinition(
        name="DisplayScatterPlot",
        display_name="DisplayScatterPlot",
        category="Data Tools",
        description="Plot a scatter plot of two measurements.",
        parameters=(
            ModuleParameter(
                "Type of measurement to plot on the X-axis",
                "Object",
                choices=("Image", "Object"),
            ),
            ModuleParameter("Select the object to plot on the X-axis", "Nuclei"),
        ),
    ),
    ModuleDefinition(
        name="DisplayDensityPlot",
        display_name="DisplayDensityPlot",
        category="Data Tools",
        description="Plot a density plot of two measurements.",
        parameters=(
            ModuleParameter(
                "Type of measurement to plot on the X-axis",
                "Object",
                choices=("Image", "Object"),
            ),
            ModuleParameter("Select the object to plot on the X-axis", "Nuclei"),
        ),
    ),
    ModuleDefinition(
        name="DisplayPlatemap",
        display_name="DisplayPlatemap",
        category="Data Tools",
        description="Display a measurement as a plate heatmap.",
        parameters=(
            ModuleParameter(
                "Display measurements of type",
                "Image",
                choices=("Image", "Object"),
            ),
            ModuleParameter("Select the measurement to display", "None"),
        ),
    ),
    # ---- Other ------------------------------------------------------------
    ModuleDefinition(
        name="DefineGrid",
        display_name="DefineGrid",
        category="Other",
        description="Define a grid for downstream grid-based identification.",
        parameters=(
            ModuleParameter("Name the grid", "Grid"),
            ModuleParameter("Number of rows", "8"),
            ModuleParameter("Number of columns", "12"),
        ),
    ),
    # ---- Worm Toolbox -----------------------------------------------------
    ModuleDefinition(
        name="IdentifyDeadWorms",
        display_name="IdentifyDeadWorms",
        category="Worm Toolbox",
        description="Identify dead (straight) worms in an image.",
        parameters=(
            ModuleParameter("Select the input image", "DNA"),
            ModuleParameter("Name the dead worm objects to be identified", "DeadWorms"),
            ModuleParameter("Worm width", "10"),
            ModuleParameter("Worm length", "100"),
        ),
    ),
    ModuleDefinition(
        name="UntangleWorms",
        display_name="UntangleWorms",
        category="Worm Toolbox",
        description="Untangle clusters of overlapping worms.",
        parameters=(
            ModuleParameter("Select the input binary image", "BinaryWorms"),
            ModuleParameter("Overlap style", "Both", choices=("Both", "With overlap", "Without overlap")),
        ),
    ),
    ModuleDefinition(
        name="StraightenWorms",
        display_name="StraightenWorms",
        category="Worm Toolbox",
        description="Straighten worms into a standard pose for measurement.",
        parameters=(
            ModuleParameter("Select the input untangled worm objects", "OverlappingWorms"),
            ModuleParameter("Name the output straightened worm objects", "StraightenedWorms"),
            ModuleParameter("Worm width", "20"),
        ),
    ),
)


def list_modules(*, category: str | None = None) -> list[ModuleDefinition]:
    """Return catalog modules, optionally filtered by category."""
    if category is None:
        return list(_MODULES)
    category_lower = category.lower()
    return [module for module in _MODULES if module.category.lower() == category_lower]


def search_modules(query: str) -> list[ModuleDefinition]:
    """Search modules by name, category, description, or parameter name."""
    query_lower = query.strip().lower()
    if not query_lower:
        return list_modules()
    matches: list[tuple[int, ModuleDefinition]] = []
    for module in _MODULES:
        searchable_fields = (
            module.name,
            module.display_name,
            module.category,
            module.description,
            " ".join(parameter.name for parameter in module.parameters),
        )
        for score, field in enumerate(searchable_fields):
            if query_lower in field.lower():
                matches.append((score, module))
                break
    return [module for _, module in sorted(matches, key=lambda item: item[0])]


def get_module_definition(name: str) -> ModuleDefinition:
    """Return a catalog module by exact name."""
    for module in _MODULES:
        if module.name == name:
            return module
    raise KeyError(f"Unknown CellProfiler module: {name}")


CATEGORY_ORDER: tuple[str, ...] = (
    "Input",
    "Image Processing",
    "Object Processing",
    "Measurement",
    "File Processing",
    "Data Tools",
    "Advanced",
    "Worm Toolbox",
    "Other",
)


def _category_sort_key(category: str) -> tuple[int, str]:
    try:
        return (CATEGORY_ORDER.index(category), "")
    except ValueError:
        return (len(CATEGORY_ORDER), category)


def list_categories(*, include_input: bool = True) -> list[str]:
    """Return catalog categories in CellProfiler-like display order."""
    categories = {module.category for module in _MODULES}
    if not include_input:
        categories.discard("Input")
    return sorted(categories, key=_category_sort_key)


def list_modules_by_category(
    *, include_input: bool = False,
) -> list[tuple[str, list[ModuleDefinition]]]:
    """Group modules by category in CellProfiler-like display order.

    The four ``Input`` setup modules are excluded by default because they are
    always present in the pipeline and are not added from the catalog.
    """
    grouped: dict[str, list[ModuleDefinition]] = {}
    for module in _MODULES:
        if not include_input and module.category == "Input":
            continue
        grouped.setdefault(module.category, []).append(module)
    for modules in grouped.values():
        modules.sort(key=lambda module: module.name.lower())
    return [
        (category, grouped[category])
        for category in sorted(grouped, key=_category_sort_key)
    ]
