let filesDir1 = [];
let filesDir2 = [];
let currentIndex = 0;

document.getElementById('dir1-input').addEventListener('change', (e) => handleFiles(e, 'dir1'));
document.getElementById('dir2-input').addEventListener('change', (e) => handleFiles(e, 'dir2'));
document.getElementById('prev-button').addEventListener('click', showPrevious);
document.getElementById('next-button').addEventListener('click', showNext);

function handleFiles(e, dir) {
    const files = Array.from(e.target.files);
    const dirName = extractDirectoryName(files[0].webkitRelativePath);

    if (dir === 'dir1') {
        filesDir1 = files;
        document.getElementById('dir1-name').textContent = `ディレクトリ: ${dirName}`;
        if (filesDir1.length > 0) loadPDF(filesDir1[0], 'pdf-viewer-dir1');
    } else {
        filesDir2 = files;
        document.getElementById('dir2-name').textContent = `ディレクトリ: ${dirName}`;
        if (filesDir2.length > 0) loadPDF(filesDir2[0], 'pdf-viewer-dir2');
    }
}

function extractDirectoryName(path) {
    const pathParts = path.split('/');
    return pathParts.slice(0, -1).join('/');
}

function showPrevious() {
    if (currentIndex > 0) {
        currentIndex--;
        if (filesDir1.length > currentIndex) loadPDF(filesDir1[currentIndex], 'pdf-viewer-dir1');
        if (filesDir2.length > currentIndex) loadPDF(filesDir2[currentIndex], 'pdf-viewer-dir2');
    }
}

function showNext() {
    if (currentIndex < Math.max(filesDir1.length, filesDir2.length) - 1) {
        currentIndex++;
        if (filesDir1.length > currentIndex) loadPDF(filesDir1[currentIndex], 'pdf-viewer-dir1');
        if (filesDir2.length > currentIndex) loadPDF(filesDir2[currentIndex], 'pdf-viewer-dir2');
    }
}

function loadPDF(file, viewerId) {
    const reader = new FileReader();
    reader.onload = function(e) {
        document.getElementById(viewerId).src = e.target.result;
    };
    reader.readAsDataURL(file);
}
