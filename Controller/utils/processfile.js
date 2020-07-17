const fs = require('fs')
const path = require('path')

function getTableData(tableData, filePath) {
    let fileContent = fs.readFileSync(filePath);
    fileContent.toString().split(/[\s?\n]/).forEach(function (item, index, array) {
        let itemArray = item.split(",");
        if (+itemArray[3] || +itemArray[4] || itemArray[5] || itemArray[6]) {
            if (+itemArray[3]) {
                tableData.push({
                    timeStamp: itemArray[0] + "秒",
                    carID: itemArray[1],
                    illegalInfo: "车辆越线"
                });
            }
            if (+itemArray[4]) {
                tableData.push({
                    timeStamp: itemArray[0] + "秒",
                    carID: itemArray[1],
                    illegalInfo: "车辆未礼让行人"
                }
                );
            } if (+itemArray[5]) {
                tableData.push({
                    timeStamp: itemArray[0] + "秒",
                    carID: itemArray[1],
                    illegalInfo: "车辆没有按照标志行驶"
                }
                );
            } if (+itemArray[6]) {
                tableData.push({
                    timeStamp: itemArray[0] + "秒",
                    carID: itemArray[1],
                    illegalInfo: "车辆闯红灯"
                }
                );
            }
        }
    })
}
module.exports = getTableData;