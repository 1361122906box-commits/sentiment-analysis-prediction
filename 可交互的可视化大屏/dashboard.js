// 初始化ECharts实例
let trendChart = echarts.init(document.getElementById('trendChart'));
let sentimentChart = echarts.init(document.getElementById('sentimentChart'));
let wordCloudChart = echarts.init(document.getElementById('wordCloudChart'));
let predictionsChart = echarts.init(document.getElementById('predictionsChart'));

// 事件数据 - 根据你的文件列表
const events = [
    {
        id: 'event1',
        name: 'A股半日成交1.25万亿缩量711亿',
        visualFile: 'A股半日成交1.25万亿缩量711亿_可视化数据.json',
        resultFile: 'A股半日成交1.25万亿缩量711亿_情感分析结果.json'
    },
    {
        id: 'event2',
        name: '中美贸易谈判',
        visualFile: '中美贸易谈判_可视化数据.json',
        resultFile: '中美贸易谈判_情感分析结果.json'
    },
    {
        id: 'event3',
        name: '公安机关查处网络谣言',
        visualFile: '公安机关查处网络谣言_可视化数据.json',
        resultFile: '公安机关查处网络谣言_情感分析结果.json'
    },
    {
        id: 'event4',
        name: '福建舰入列',
        visualFile: '福建舰入列_可视化数据.json',
        resultFile: '福建舰入列_情感分析结果.json'
    },
    {
        id: 'event5',
        name: '经济政策解读',
        visualFile: '经济政策解读_可视化数据.json',
        resultFile: '经济政策解读_情感分析结果.json'
    },
    {
        id: 'event6',
        name: '泡泡玛特直播事故',
        visualFile: '泡泡玛特直播事故_可视化数据.json',
        resultFile: '泡泡玛特直播事故_情感分析结果.json'
    }
];

// 当前选中的事件
let currentEvent = null;
let currentVisualData = null;
let currentResultData = null;

// 初始化页面
async function initDashboard() {
    // 加载事件列表
    loadEventList();

    // 默认加载第一个事件
    if (events.length > 0) {
        await loadEventData(events[0]);
    }
}

// 加载事件列表
function loadEventList() {
    const eventList = document.getElementById('eventList');
    eventList.innerHTML = '';

    events.forEach((event, index) => {
        const li = document.createElement('li');
        li.className = 'event-item';
        if (index === 0) li.classList.add('active');

        li.innerHTML = `
            <div class="event-name">${event.name}</div>
            <div class="event-stats">
                <span class="sentiment-badge positive">积极 0%</span>
                <span class="sentiment-badge negative">消极 0%</span>
                <span class="sentiment-badge neutral">中性 0%</span>
            </div>
        `;

        li.addEventListener('click', async () => {
            // 更新选中状态
            document.querySelectorAll('.event-item').forEach(item => {
                item.classList.remove('active');
            });
            li.classList.add('active');

            // 加载事件数据
            await loadEventData(event);
        });

        eventList.appendChild(li);
    });
}

// 加载事件数据
async function loadEventData(event) {
    currentEvent = event;

    try {
        // 加载可视化数据
        const visualResponse = await fetch(event.visualFile);
        currentVisualData = await visualResponse.json();

        // 加载情感分析结果数据
        const resultResponse = await fetch(event.resultFile);
        currentResultData = await resultResponse.json();

        // 更新所有图表和数据
        updateAllCharts();

    } catch (error) {
        console.error('加载数据失败:', error);
        alert('数据加载失败，请检查文件路径和格式');
    }
}

// 更新所有图表
function updateAllCharts() {
    if (!currentVisualData) return;

    updateSentimentChart();
    updateTrendChart();
    updateWordCloudChart();
    updatePredictionsChart();
    updateCommentsList();
    updateSummary();
    updateEventListStats();
}

// 更新情感分布图
function updateSentimentChart() {
    const distribution = currentVisualData.sentiment_distribution;

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'item',
            formatter: '{b}: {c} ({d}%)'
        },
        legend: {
            bottom: '0%',
            data: distribution.categories,
            textStyle: {
                color: '#a0aec0',
                fontSize: 12
            }
        },
        series: [{
            name: '情感分布',
            type: 'pie',
            radius: ['45%', '70%'],
            center: ['50%', '45%'],
            avoidLabelOverlap: false,
            itemStyle: {
                borderRadius: 8,
                borderColor: '#1a2b45',
                borderWidth: 2
            },
            label: {
                show: false
            },
            emphasis: {
                label: {
                    show: true,
                    fontSize: 14,
                    fontWeight: 'bold',
                    color: '#fff'
                }
            },
            data: distribution.categories.map((cat, index) => ({
                value: distribution.values[index],
                name: cat,
                itemStyle: { color: distribution.colors[index] }
            }))
        }]
    };

    sentimentChart.setOption(option);
}

// 更新时间趋势图
function updateTrendChart() {
    const timeSeries = currentVisualData.time_series || [];

    // 处理时间数据
    const times = timeSeries.map(item => {
        const timeStr = item.time;
        // 提取小时部分
        const match = timeStr.match(/\d{2}:\d{2}/);
        return match ? match[0] : timeStr.split(' ')[1];
    });

    const scores = timeSeries.map(item => item.sentiment_score);

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const time = params[0].axisValue;
                const score = params[0].data;
                let sentiment = '中性';
                let color = '#faad14';

                if (score > 0.2) {
                    sentiment = '积极';
                    color = '#52c41a';
                } else if (score < -0.2) {
                    sentiment = '消极';
                    color = '#ff4d4f';
                }

                return `
                    <div style="color: ${color}">
                        <strong>${time}</strong><br/>
                        情感得分: ${score.toFixed(3)}<br/>
                        情感倾向: ${sentiment}
                    </div>
                `;
            }
        },
        xAxis: {
            type: 'category',
            data: times,
            axisLine: {
                lineStyle: { color: '#4a5568' }
            },
            axisLabel: {
                color: '#a0aec0',
                fontSize: 11,
                rotate: 45
            }
        },
        yAxis: {
            type: 'value',
            axisLine: {
                lineStyle: { color: '#4a5568' }
            },
            axisLabel: {
                color: '#a0aec0',
                fontSize: 11
            },
            splitLine: {
                lineStyle: {
                    color: '#2d3748',
                    type: 'dashed'
                }
            }
        },
        series: [{
            data: scores,
            type: 'line',
            smooth: true,
            symbol: 'circle',
            symbolSize: 6,
            lineStyle: {
                color: '#00d4ff',
                width: 2
            },
            itemStyle: {
                color: '#00d4ff'
            },
            areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: 'rgba(0, 212, 255, 0.3)' },
                    { offset: 1, color: 'rgba(0, 212, 255, 0.05)' }
                ])
            },
            markLine: {
                silent: true,
                data: [
                    { yAxis: 0.2, lineStyle: { color: '#52c41a', type: 'dashed', width: 1 } },
                    { yAxis: -0.2, lineStyle: { color: '#ff4d4f', type: 'dashed', width: 1 } },
                    { yAxis: 0, lineStyle: { color: '#faad14', type: 'dashed', width: 1 } }
                ],
                label: { show: false }
            }
        }],
        grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
        }
    };

    trendChart.setOption(option);
}

// 更新词云图
function updateWordCloudChart() {
    const keywords = currentVisualData.keywords || {};

    // 转换关键词数据格式
    const wordData = Object.entries(keywords).map(([name, value]) => ({
        name: name,
        value: value
    })).sort((a, b) => b.value - a.value).slice(0, 50); // 取前50个

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            show: true,
            formatter: function(params) {
                return `${params.name}: ${params.value}次`;
            }
        },
        series: [{
            type: 'wordCloud',
            shape: 'circle',
            left: 'center',
            top: 'center',
            width: '95%',
            height: '95%',
            sizeRange: [14, 40],
            rotationRange: [0, 0],
            rotationStep: 45,
            gridSize: 10,
            drawOutOfBound: false,
            textStyle: {
                fontFamily: 'Microsoft YaHei',
                fontWeight: 'bold',
                color: function () {
                    const colors = [
                        '#4ecdc4', '#44af69', '#f8c630', '#f24236',
                        '#5dade2', '#af7ac5', '#f39c12', '#e74c3c'
                    ];
                    return colors[Math.floor(Math.random() * colors.length)];
                }
            },
            emphasis: {
                focus: 'self',
                textStyle: {
                    shadowBlur: 10,
                    shadowColor: '#333'
                }
            },
            data: wordData
        }]
    };

    wordCloudChart.setOption(option);
}

// 更新预测图表
function updatePredictionsChart() {
    const predictions = currentVisualData.predictions || [];

    if (predictions.length === 0) {
        predictionsChart.setOption({
            backgroundColor: 'transparent',
            title: {
                text: '暂无预测数据',
                left: 'center',
                top: 'center',
                textStyle: {
                    color: '#a0aec0',
                    fontSize: 14
                }
            }
        });
        return;
    }

    const times = predictions.map(item => {
        const timeStr = item.time;
        const match = timeStr.match(/\d{2}:\d{2}/);
        return match ? match[0] : timeStr.split(' ')[1];
    });

    const scores = predictions.map(item => item.sentiment_score);

    const option = {
        backgroundColor: 'transparent',
        tooltip: {
            trigger: 'axis',
            formatter: function(params) {
                const time = params[0].axisValue;
                const score = params[0].data;
                return `预测时间: ${time}<br/>预测得分: ${score.toFixed(3)}`;
            }
        },
        xAxis: {
            type: 'category',
            data: times,
            axisLine: {
                lineStyle: { color: '#4a5568' }
            },
            axisLabel: {
                color: '#a0aec0',
                fontSize: 11,
                rotate: 45
            }
        },
        yAxis: {
            type: 'value',
            axisLine: {
                lineStyle: { color: '#4a5568' }
            },
            axisLabel: {
                color: '#a0aec0',
                fontSize: 11
            },
            splitLine: {
                lineStyle: {
                    color: '#2d3748',
                    type: 'dashed'
                }
            }
        },
        series: [{
            data: scores,
            type: 'line',
            smooth: true,
            symbol: 'circle',
            symbolSize: 6,
            lineStyle: {
                color: '#9b59b6',
                width: 2
            },
            itemStyle: {
                color: '#9b59b6'
            },
            areaStyle: {
                color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: 'rgba(155, 89, 182, 0.3)' },
                    { offset: 1, color: 'rgba(155, 89, 182, 0.05)' }
                ])
            }
        }],
        grid: {
            left: '3%',
            right: '4%',
            bottom: '15%',
            containLabel: true
        }
    };

    predictionsChart.setOption(option);
}

// 更新评论列表
function updateCommentsList() {
    const commentsList = document.getElementById('commentsList');
    commentsList.innerHTML = '';

    if (!currentResultData || !Array.isArray(currentResultData)) {
        commentsList.innerHTML = '<div class="comment-item">暂无评论数据</div>';
        return;
    }

    // 显示最近的20条评论
    const recentComments = currentResultData.slice(0, 20);

    recentComments.forEach(comment => {
        const div = document.createElement('div');
        const sentiment = comment.sentiment;
        let sentimentClass = 'comment-neutral';
        let sentimentTagClass = 'neutral-tag';

        if (sentiment === '正面') {
            sentimentClass = 'comment-positive';
            sentimentTagClass = 'positive-tag';
        } else if (sentiment === '负面') {
            sentimentClass = 'comment-negative';
            sentimentTagClass = 'negative-tag';
        }

        // 格式化时间
        const timestamp = comment.timestamp;
        const displayTime = timestamp ? timestamp.split(' ')[1] : '未知时间';

        // 截断过长的评论
        const content = comment.comment.length > 100
            ? comment.comment.substring(0, 100) + '...'
            : comment.comment;

        div.className = `comment-item ${sentimentClass}`;
        div.innerHTML = `
            <div class="comment-header">
                <span class="sentiment-tag ${sentimentTagClass}">${sentiment}</span>
                <span class="comment-time">${displayTime}</span>
            </div>
            <div class="comment-content">${content}</div>
        `;

        commentsList.appendChild(div);
    });
}

// 更新舆情总结
function updateSummary() {
    const summaryText = document.getElementById('summaryText');

    if (currentVisualData.event_info) {
        const info = currentVisualData.event_info;
        summaryText.innerHTML = `
            <strong>${info.name}</strong><br/>
            分析时间: ${info.analysis_time}<br/>
            总评论数: ${info.total_comments}条<br/>
            情感总结: ${info.sentiment_summary}<br/>
            积极比例: ${info.positive_percentage}% | 
            消极比例: ${info.negative_percentage}% | 
            中性比例: ${info.neutral_percentage}%
        `;
    } else {
        summaryText.textContent = '暂无舆情总结信息';
    }
}

// 更新事件列表中的统计数据
function updateEventListStats() {
    if (!currentVisualData.event_info) return;

    const info = currentVisualData.event_info;
    const eventItems = document.querySelectorAll('.event-item');

    eventItems.forEach(item => {
        const eventName = item.querySelector('.event-name').textContent;
        if (eventName === currentEvent.name) {
            const statsDiv = item.querySelector('.event-stats');
            statsDiv.innerHTML = `
                <span class="sentiment-badge positive">积极 ${info.positive_percentage}%</span>
                <span class="sentiment-badge negative">消极 ${info.negative_percentage}%</span>
                <span class="sentiment-badge neutral">中性 ${info.neutral_percentage}%</span>
            `;
        }
    });
}

// 时间过滤器点击事件
document.getElementById('timeFilter').addEventListener('click', function() {
    const current = this.textContent;
    const options = ['24H', '7D', '30D', '全部'];
    const currentIndex = options.indexOf(current);
    const nextIndex = (currentIndex + 1) % options.length;
    this.textContent = options[nextIndex];

    // 这里可以添加根据时间范围过滤数据的逻辑
    console.log('切换时间范围:', options[nextIndex]);
});

// 窗口大小变化时重绘图表
window.addEventListener('resize', function() {
    trendChart.resize();
    sentimentChart.resize();
    wordCloudChart.resize();
    predictionsChart.resize();
});

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', initDashboard);