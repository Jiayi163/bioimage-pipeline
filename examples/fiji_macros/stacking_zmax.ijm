// Parameterized version of Stacking+Drectly.ijm
// Usage: fiji-windows-x64.exe --headless -macro stacking_zmax.ijm <input_dir> <output_dir>
// Arguments must be absolute directory paths (trailing separator recommended).

input = ensureTrailingSeparator(getArgument());
output = ensureTrailingSeparator(getArgument());

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
            print("Skipping non-image file: " + dir + name);
        }
    }
}

function processFile(inputFolder, outputFolder, file) {
    inputFolder = ensureTrailingSeparator(inputFolder);
    outputFolder = ensureTrailingSeparator(outputFolder);
    inputPath = inputFolder + file;
    if (!File.exists(inputPath)) {
        print("File not found: " + inputPath);
        return;
    }
    if (!File.exists(outputFolder)) {
        File.makeDirectory(outputFolder);
    }

    // Windowless Bio-Formats import avoids GUI dialogs that break headless runs.
    importOptions = "open=[" + inputPath + "] autoscale view=Hyperstack stack_format=Default";
    print("Bio-Formats macro command: run(\"Bio-Formats Windowless Importer\", \"" + importOptions + "\");");
    run("Bio-Formats Windowless Importer", importOptions);
    if (nImages == 0) {
        print("Import failed: " + inputPath);
        return;
    }
    selectImage(1);
    run("Z Project...", "projection=[Max Intensity]");
    saveName = replace(file, ".oir", ".tif");
    saveAs("Tiff", outputFolder + saveName);
    close("*");
}

processFolder(input);
setBatchMode(false);
