const ffmpeg = require('fluent-ffmpeg');
const fs = require('fs');
const path = require('path');
const setting = require('../setting');
const { stderr } = require('process');
const child_process = require('child_process');

function changeToMp4(filePath, outputPath, res) {
    let ffmpegExePath = setting.FFMPEG_PATH;
    let command = ffmpegExePath + " " + "-i" + " " + filePath + " " + "-c:v" + " " + "libx264" + " " + outputPath;
    child_process.execSync(command);
    fs.unlinkSync(filePath);
    res.send("video upload success");
}

function ifH264(filePath) {
    let ffmpegExePath = setting.FFMPEG_PATH;
    let reg = /h264/;
    let command = ffmpegExePath + " " + "-i" + " " + filePath + " " + "-hide_banner"
    child_process.exec(command, (error, stdout, stderr) => {
        if (reg.test(error.message)) {
            return true;
        }
    })
}

module.exports = { changeToMp4, ifH264 }