import requests
import os
import re
from model.res import Response
def parse_ip_info(text):
    """解析返回文本，提取IP、省份、城市"""
    # 提取IP
    ip_match = re.search(r'(\d+\.\d+\.\d+\.\d+)', text)
    ip = ip_match.group(1) if ip_match else None
    
    # 提取省份和城市（格式：中国 河北 石家庄）
    # 匹配 "中国" 后面的两个词
    area_match = re.search(r'中国\s+([^\s]+)\s+([^\s]+)', text)
    if area_match:
        province = area_match.group(1)
        city = area_match.group(2)
        return {
            'ip': ip,
            'province': province,
            'city': city
        }
    
    return {'ip': ip}
def get_current_city_by_ip() -> str:
    """
    通过当前服务器的公网IP，自动获取所在城市。
    不需要输入参数。
    """
    try:
        # 使用 ip-api.com 的免费接口（国内访问稳定）
        res = requests.get('http://myip.ipip.net', timeout=10).text
        ip_info = parse_ip_info(res)
       
        return ip_info['city']
    except Exception as e:
        return f"IP定位失败: {str(e)}"

def get_city_by_ip(ip=None):
    """
    根据 IP 地址获取城市信息
    ip: 可选，不传则自动获取当前请求的 IP
    """
    url = 'https://restapi.amap.com/v3/ip'
    params = {
        'key': os.getenv('AMAP_API_KEY')
    }
    if ip:
        params['ip'] = ip
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if data.get('status') == '1':
            return {
                'province': data.get('province'),    # 省份
                'city': data.get('city'),            # 城市
                'adcode': data.get('adcode'),        # 行政区划代码
                'ip': data.get('ip')                 # 查询的 IP
            }
        else:
            print(f"API错误: {data.get('info')}")
            
    except Exception as e:
        print(f"请求失败: {e}")
    
    return None

def get_citycode_by_city(city: str) -> object:
    """
    通过城市名，获取城市编码。
    :param city: 城市名
    :return: 城市编码
    """
    try:
        res = requests.get(f'https://restapi.amap.com/v3/config/district?keywords={city}&subdistrict=0&key={os.getenv("AMAP_API_KEY")}', timeout=10).json()
        
        if res.get('district'):
            return res['district'][0]['adcode']
        else:
            return f"无法获取{city}的编码"
    except Exception as e:
        return f"获取{city}的编码失败: {str(e)}"



def get_weather_by_city(city: str) -> str:
    """
    通过城市名，获取天气信息。
    :return: 天气信息
    """
    try:

        res = requests.get(f'https://restapi.amap.com/v3/weather/weatherInfo?key={os.getenv("AMAP_API_KEY")}&city={city}&extensions=all', timeout=10).json()

        if res.get('infocode') == '10000':
            return res
        else:
            raise Exception(f"高德获取天气信息错误,get_weather_by_city1")
    except Exception as e:
        raise Exception(f"高德获取天气信息错误,get_weather_by_city2")

from langchain.tools import tool
@tool
def get_weather()-> Response:
    """获取当前公网城市的天气"""
    try:
        city = get_current_city_by_ip()
        return Response.ok(data=get_weather_by_city(city))
    except Exception as e:
        return Response.error(message=f"获取当前公网城市的天气失败: {str(e)}")

