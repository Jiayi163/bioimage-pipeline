// Batch export CellProfiler TIFF outputs into masks/ and labels/ folders.
// Arguments are joined by "|" from Python:
//   input_dir|masks_dir|labels_dir|image_pattern

args = split(getArgument(), "|");
inputDir = ensureTrailingSlash(args[0]);
masksDir = ensureTrailingSlash(args[1]);
labelsDir = ensureTrailingSlash(args[2]);
pattern = "*.tif";
if (args.length >= 4) {
    pattern = args[3];
}

File.makeDirectory(masksDir);
File.makeDirectory(labelsDir);
processFolder(inputDir);

function ensureTrailingSlash(path) {
    path = replace(path, "\\", "/");
    if (!endsWith(path, "/")) {
        path = path + "/";
    }
    return path;
}

function processFolder(folder) {
    list = getFileList(folder);
    for (i = 0; i < list.length; i++) {
        path = folder + list[i];
        if (File.isDirectory(path)) {
            processFolder(ensureTrailingSlash(path));
        } else if (matchesPattern(list[i], pattern)) {
            exportTiff(path, list[i]);
        }
    }
}

function matchesPattern(name, pattern) {
    lower = toLowerCase(name);
    if (endsWith(lower, ".tif") || endsWith(lower, ".tiff")) {
        return true;
    }
    return false;
}

function exportTiff(path, name) {
    lower = toLowerCase(name);
    open(path);
    if (indexOf(lower, "mask") >= 0 && indexOf(lower, "labeled") < 0) {
        saveAs("Tiff", masksDir + name);
    } else if (indexOf(lower, "label") >= 0 || indexOf(lower, "object") >= 0 || indexOf(lower, "segmented") >= 0) {
        saveAs("Tiff", labelsDir + name);
    }
    close();
}
