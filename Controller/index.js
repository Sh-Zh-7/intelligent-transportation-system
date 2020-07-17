const express = require('express');
const path = require('path');
const fs = require('fs');
const child_process = require('child_process');
const multer = require('multer')
const getTableData = require('./utils/processfile.js');
const changToMp4 = require('./utils/format.js').changeToMp4;
const ifH264 = require('./utils/format.js').ifH264;
const setting = require('./setting.js')

const app = express();

const upload = multer({ dest: './public/UploadFiles/' })

const history = require('connect-history-api-fallback');

app.use(history({
    verbose: true,
    htmlAcceptHeaders: ['text/html', 'application/xhtml+xml']
}));

app.use(express.static(path.join(__dirname, 'dist')));
app.use(express.static(path.join(__dirname, 'public')));

app.all('*', function (req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "X-Requested-With");
    res.header("Access-Control-Allow-Methods", "PUT,POST,GET,DELETE,OPTIONS");
    res.header("X-Powered-By", ' 3.2.1')
    res.header("Content-Type", "application/json;charset=utf-8");
    next();
});


app.listen(8000, () => {
    console.log("server start");
})

app.post('/upload/video', upload.single("video"), (req, res) => {
    let ext = req.file.originalname.split('.')[1]
    if (ext === "avi") {
        let newpath = path.join(path.parse(req.file.path).dir, "upload_video.mp4");
        changToMp4(req.file.path, newpath, res);
    }
    else if (ext === "mp4") {
        if (ifH264(req.file.path)) {
            let newpath = path.join(path.parse(req.file.path).dir, "upload_video.mp4");
            fs.renameSync(req.file.path, newpath);
            res.send("video upload success");
        } else {
            let newpath = path.join(path.parse(req.file.path).dir, "upload_video.mp4");
            changToMp4(req.file.path, newpath, res);
        }
    }
})

app.post('/upload/image', upload.single('image'), (req, res) => {
    let ext = req.file.originalname.split('.')[1]
    let newFullName = "upload_image." + ext;
    let newpath = path.join(path.parse(req.file.path).dir, newFullName)
    fs.renameSync(req.file.path, newpath);
    res.send("image upload success");
})

app.get('/api/process', (req, res) => {
    /* let command = "";
    let reg = new RegExp("(.jpg|.png|.jpeg)$");
    let pythonPath = setting.PYTHONEXE_PATH;
    let pythonFilePath = setting.PYTHONFILE_PATH;
    let inputVideo = path.join(__dirname, "public", "UploadFiles", "upload_video.mp4");
    let imageName;
    let fileList = fs.readdirSync(path.join(__dirname, "public", "UploadFiles"));
    fileList.forEach((item, index) => {
        if (reg.test(item)) {
            imageName = item;
            console.log(item);
        }
    })
    let inputImage = path.join(__dirname, "public", "UploadFiles", imageName);
    let outputDir = path.join(__dirname, "public", "OutputFiles");
    command += pythonPath + " " + pythonFilePath + " " + "--input_video" + " " + inputVideo + " " + "--input_background" + " " + inputImage + " " + "--output_dir" + " " + outputDir;
    console.log("command: " + command);
    child_process.execSync(command);
    res.json({
        message: 'process finish'
    }) */
    setTimeout(() => {
        res.json({
            message: "process finish"
        })
    }, 3 * 1000);
})

app.get('/api/videodata', (req, res) => {
    let resultPath = path.join(__dirname, "public", "OutputFiles", "result.txt");
    let tableData = [];
    getTableData(tableData, resultPath);
    res.json({
        videodata: tableData
    })
})

app.get("/api/downloadvideo", (req, res) => {
    let filePath = path.join(__dirname, "public", "OutputFiles", "output_video.mp4");
    res.download(filePath, 'video.mp4')
    res.set({
        "Content-Type": "application/octet-stream",
    })
})

app.get("/api/downloadtxt", (req, res) => {
    let filePath = path.join(__dirname, "public", "OutputFiles", "result.txt");
    res.download(filePath, 'result.txt')
    res.set({
        "Content-Type": "application/octet-stream",
    })
})

app.get("/api/cleardir", (req, res) => {
    let uploadPath = path.join(__dirname, "public", "UploadFiles");
    let fileList = fs.readdirSync(uploadPath)
    fileList.forEach((item, index) => {
        fs.unlinkSync(path.join(uploadPath, item))
    })
    let outputPath = path.join(__dirname, "public", "OutputFiles");
    let outputFileList = fs.readdirSync(outputPath)
    outputFileList.forEach((item, index) => {
        fs.unlinkSync(path.join(outputPath, item))
    })
    res.status(200).end();
})
