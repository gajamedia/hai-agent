// A picture (a design to read), as opposed to a table file (CSV, Excel). By name here; the server checks the content.
export const isImageFile = (file) => /\.(png|jpe?g|webp)$/i.test(file?.name || "");
