// Z-max project one multipage TIFF stack (no Bio-Formats import).
// Usage: fiji-windows-x64.exe --headless -macro stacking_zmax_single_tiff.ijm <stack.tif> <output.tif>

input = getArgument();
output = getArgument();

setBatchMode(true);
open(input);
if (nImages == 0) {
    print("Open failed: " + input);
    exit;
}
selectImage(1);
run("Z Project...", "projection=[Max Intensity]");
saveAs("Tiff", output);
close("*");
setBatchMode(false);
