// Parameterized version of Stacking+Drectly.ijm
// Usage: fiji-windows-x64.exe --headless -macro stacking_zmax.ijm <input_dir> <output_dir>
// Arguments must be absolute directory paths (trailing separator recommended).
// For GUI/workflow runs, prefer the generated macro with embedded paths in logs/.

input = ensureTrailingSeparator(getArgument());
output = ensureTrailingSeparator(getArgument());

print("[OIR] macro input folder: " + input);
print("[OIR] macro output folder: " + output);

setBatchMode(true);
if (!File.exists(output)) {
    File.makeDirectory(output);
}

function ensureTrailingSeparator(path) {
    if (endsWith(path, "/") || endsWith(path, "\\")) {
        return path;
    }
    return path + "/";
}

function processFolder(dir) {
    dir = ensureTrailingSeparator(dir);
    list = getFileList(dir);
    for (i = 0; i < list.length; i++) {
        name = list[i];
        lower = toLowerCase(name);
        if (endsWith(lower, ".oir")) {
            processFile(dir, output, name);
        } else if (endsWith(name, "/")) {
            processFolder(dir + name);
        } else {
            print("[OIR] skipping non-image file: " + dir + name);
        }
    }
}

function processFile(inputFolder, outputFolder, file) {
    inputFolder = ensureTrailingSeparator(inputFolder);
    outputFolder = ensureTrailingSeparator(outputFolder);
    inputPath = inputFolder + file;
    print("[OIR] input path: " + inputPath);
    print("[OIR] output folder: " + outputFolder);
    if (!File.exists(inputPath)) {
        print("[OIR] file not found: " + inputPath);
        return;
    }
    if (!File.exists(outputFolder)) {
        File.makeDirectory(outputFolder);
    }

    // Windowless Bio-Formats import avoids GUI dialogs that break headless runs.
    importOptions = "open=[" + inputPath + "] autoscale view=Hyperstack stack_format=Default";
    print("[OIR] Bio-Formats import options: " + importOptions);
    run("Bio-Formats Windowless Importer", importOptions);
    print("[OIR] nImages after Bio-Formats import: " + nImages);
    if (nImages == 0) {
        print("[OIR] import failed: " + inputPath);
        return;
    }
    selectImage(1);
    print("[OIR] current image title: " + getTitle());
    if (is("hyperstack")) {
        print("[OIR] stack slices: " + nSlices);
    } else {
        print("[OIR] stack slices: n/a (not a hyperstack)");
    }
    run("Z Project...", "projection=[Max Intensity]");
    print("[OIR] Z Project completed");
    saveName = replace(file, ".oir", ".tif");
    savePath = outputFolder + saveName;
    print("[OIR] saveAs target: " + savePath);
    saveAs("Tiff", savePath);
    print("[OIR] saveAs called");
    if (File.exists(savePath)) {
        print("[OIR] saved file verified on disk: " + savePath);
    } else {
        print("[OIR] WARNING: saved file not found on disk: " + savePath);
    }
    close("*");
}

processFolder(input);
setBatchMode(false);
