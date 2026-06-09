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
            if parameter.visibility.is_visible(current)
        ]


_MODULES: tuple[ModuleDefinition, ...] = (
    ModuleDefinition(
        name="Images",
        display_name="Images",
        category="Input",
        description="Collect image file names for a CellProfiler pipeline.",
        parameters=(
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
            ),
            ModuleParameter(
                "Extraction method count",
                "1",
                "Number of metadata extraction methods configured.",
                visibility=visible_when("Extract metadata?", "Yes"),
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
            ),
            ModuleParameter(
                "Single images count",
                "0",
                "Number of single-image assignments.",
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
            ),
            ModuleParameter(
                "Name to assign these objects",
                "Cell",
                "Object name placeholder used by object-image assignments.",
            ),
            ModuleParameter(
                "Select the image type",
                "Grayscale image",
                "Per-assignment image type.",
                ("Grayscale image", "Color image", "Mask", "Illumination function"),
            ),
            ModuleParameter(
                "Set intensity range from",
                "Image metadata",
                "Per-assignment intensity range source.",
                ("Image metadata", "Manual"),
            ),
            ModuleParameter("Maximum intensity", "255.0"),
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
            ModuleParameter("Threshold setting version", "11"),
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
        category="Data Tools",
        description="Classify objects using CellProfiler measurements.",
        parameters=(ModuleParameter("Select the object to be classified", "Nuclei"),),
    ),
    ModuleDefinition(
        name="SaveImages",
        display_name="SaveImages",
        category="Output",
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
        category="Output",
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
            ModuleParameter("Press button to select measurements", "None|None"),
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


def list_categories() -> list[str]:
    """Return sorted catalog categories."""
    return sorted({module.category for module in _MODULES})
