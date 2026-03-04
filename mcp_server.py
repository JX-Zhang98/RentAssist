"""
租房仿真 API MCP Server (stdio 模式)
将 AgentGameFakeAppApi 中的 15 个接口 + 1 个 init 接口封装为 MCP 工具
"""

import os
import json
import re
from pathlib import Path
import httpx
from mcp.server.fastmcp import FastMCP

# 从 config.json 读取配置
_config_path = Path(__file__).parent / "config.json"
with open(_config_path, "r", encoding="utf-8") as f:
    _config = json.load(f)

API_BASE_URL = _config.get("apiaddr", "http://localhost:8080")
DEFAULT_USER_ID = _config.get("userid", "")

mcp = FastMCP("RentAssist", instructions="北京租房仿真 API 工具集")

# ==================== 缓存 ====================
_CACHE_DIR = Path(__file__).parent / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _cache_filename(path: str, params: dict | None, user_id: str | None) -> str:
    """用路径和参数明文拼接缓存文件名"""
    # /api/houses/by_platform -> houses_by_platform
    name = path.strip("/").replace("/", "_").replace("api_", "")
    parts = [name]
    for k, v in sorted((params or {}).items()):
        parts.append(f"{k}={v}")
    filename = "__".join(parts)
    # 去掉文件名中不安全的字符
    filename = re.sub(r'[^\w=.,\-]', '_', filename)
    return filename + ".json"


def _cache_get(path: str, params: dict | None, user_id: str | None) -> str | None:
    fp = _CACHE_DIR / _cache_filename(path, params, user_id)
    if fp.exists():
        return fp.read_text(encoding="utf-8")
    return None


def _cache_set(path: str, params: dict | None, user_id: str | None, value: str) -> None:
    fp = _CACHE_DIR / _cache_filename(path, params, user_id)
    value = json.dumps({"query":params, "resp":json.loads(value)}, indent=2, ensure_ascii=False)
    fp.write_text(value, encoding="utf-8")


def _headers(user_id: str | None = None) -> dict:
    uid = user_id or DEFAULT_USER_ID
    if uid:
        return {"X-User-ID": uid}
    return {}


def _build_params(**kwargs) -> dict:
    """过滤掉 None 值的参数"""
    return {k: v for k, v in kwargs.items() if v is not None}


async def _get(path: str, params: dict | None = None, user_id: str | None = None) -> str:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30) as client:
        resp = await client.get(path, params=params, headers=_headers(user_id))
        _cache_set(path, params, user_id, resp.text)
        return resp.text


async def _post(path: str, params: dict | None = None, user_id: str | None = None) -> str:
    async with httpx.AsyncClient(base_url=API_BASE_URL, timeout=30) as client:
        resp = await client.post(path, params=params, headers=_headers(user_id))
        return resp.text


# ==================== 地标接口 ====================

@mcp.tool()
async def get_landmarks(category: str | None = None, district: str | None = None) -> str:
    """查询行政区内不同类别地标列表，支持 category、district 筛选（取交集）。
    用于查地铁站、公司、商圈等地标。
    Args:
        category: 地标类别 subway(地铁)/company(公司)/landmark(商圈等)，不传则不过滤
        district: 行政区，如 海淀、朝阳
    """
    params = _build_params(category=category, district=district)
    return await _get("/api/landmarks", params)


@mcp.tool()
async def get_landmark_by_name(name: str) -> str:
    """按名称精确查询地标，如西二旗站、百度。返回地标 id、经纬度等，用于后续 nearby 查房。

    Args:
        name: 地标名称，如 西二旗站、国贸
    """
    tool_res = await _get(f"/api/landmarks/name/{name}")
    landmark = json.loads(tool_res)['data']
    landmark['type'] = landmark['details']['type']
    landmark['lines'] = landmark['details']['lines']
    del landmark['details']
    return json.dumps(tool_res)


@mcp.tool()
async def search_landmarks(keyword: str, category: str | None = None, district: str | None = None) -> str:
    """关键词模糊搜索地标。支持 category、district 同时筛选，多条件取交集。
    Args:
        keyword: 搜索关键词，必填
        category: 可选，限定类别 subway/company/landmark
        district: 可选，限定行政区，如 海淀、朝阳
    """
    params = _build_params(keyword=keyword, category=category, district=district)
    return await _get("/api/landmarks/search", params)


@mcp.tool()
async def get_landmark_by_id(id: str) -> str:
    """按地标 id 查询地标详情。
    Args:
        id: 地标 ID，如 SS_001、LM_002
    """
    return await _get(f"/api/landmarks/{id}")


@mcp.tool()
async def get_landmark_stats() -> str:
    """获取地标统计信息（总数、按类别分布等）。"""
    return await _get("/api/landmarks/stats")


# ==================== 房源接口 ====================

@mcp.tool()
async def get_house_by_id(house_id: str) -> str:
    """根据房源 ID 获取单套房源详情（安居客）。
    Args:
        house_id: 房源 ID，如 HF_2001
    """
    return await _get(f"/api/houses/{house_id}", user_id=DEFAULT_USER_ID)


@mcp.tool()
async def get_house_listings(house_id: str) -> str:
    """根据房源 ID 获取该房源在链家/安居客/58同城各平台的全部挂牌记录。
    Args:
        house_id: 房源 ID，如 HF_2001
    """
    return await _get(f"/api/houses/listings/{house_id}", user_id=DEFAULT_USER_ID)


@mcp.tool()
async def get_houses_by_community(
    community: str,
    listing_platform: str | None = None,
) -> str:
    """按小区名查询该小区下可租房源。未传 listing_platform 时只返回安居客。

    Args:
        community: 小区名，如 建清园(南区)、保利锦上(二期)
        listing_platform: 挂牌平台：链家/安居客/58同城，不传默认安居客
    """
    params = _build_params(community=community, listing_platform=listing_platform, page=1, page_size=5)
    tool_res = await _get("/api/houses/by_community", params, user_id=DEFAULT_USER_ID)
    all_houses = json.loads(tool_res)["data"]["items"]
    tool_res = json.dumps(all_houses[:5])
    
    return tool_res


@mcp.tool()
async def get_houses_by_platform(
    listing_platform: str | None = None,
    district: str | None = None,
    area: str | None = None,
    min_price: int | None = None,
    max_price: int | None = None,
    bedrooms: str | None = None,
    rental_type: str | None = None,
    decoration: str | None = None,
    orientation: str | None = None,
    elevator: str | None = None,
    min_area: int | None = None,
    max_area: int | None = None,
    property_type: str | None = None,
    subway_line: str | None = None,
    max_subway_dist: int | None = None,
    subway_station: str | None = None,
    utilities_type: str | None = None,
    available_from_before: str | None = None,
    commute_to_xierqi_max: int | None = None,
    sort_by: str | None = None,
    sort_order: str | None = None,
    normal_tags: list[str] | None = None,
    forbidden_tags: list[str] | None = None,
    
) -> str:
    params = _build_params(
        listing_platform=listing_platform, district=district, area=area,
        min_price=min_price, max_price=max_price, bedrooms=bedrooms,
        rental_type=rental_type, decoration=decoration, orientation=orientation,
        elevator=elevator, min_area=min_area, max_area=max_area,
        property_type=property_type, subway_line=subway_line,
        max_subway_dist=max_subway_dist, subway_station=subway_station,
        utilities_type=utilities_type, available_from_before=available_from_before,
        commute_to_xierqi_max=commute_to_xierqi_max,
        sort_by=sort_by, sort_order=sort_order, page=1, page_size=100,
    )
    tool_res = await _get("/api/houses/by_platform", params, user_id=DEFAULT_USER_ID)
 
    all_houses = json.loads(tool_res)["data"]["items"]
    required_tag_set = set(normal_tags or [])
    forbidden_tag_set = set(forbidden_tags or [])
    scored_houses: list[tuple[int, dict]] = []

    for house in all_houses:
        house_tags = set(house.get("tags") or [])
        if forbidden_tag_set and forbidden_tag_set.intersection(house_tags):
            continue

        overlap_count = len(required_tag_set.intersection(house_tags))
        scored_houses.append((overlap_count, house))

    # Python sort is stable, so equal overlap_count keeps original all_houses order.
    scored_houses.sort(key=lambda x: x[0], reverse=True)
    ret_houses = [house for _, house in scored_houses[:5]]

    tool_res = json.dumps(ret_houses)
    return tool_res

with open("tags.json", "r") as f:
    tags = json.load(f)
    positive_tags = tags['positive']
    negative_tags = tags['negative']
    
get_houses_by_platform.__doc__ = """查询可租房源，支持多维度筛选。这是最强大的房源搜索工具。
    Args:
        listing_platform: 挂牌平台：链家/安居客/58同城,不传默认安居客
        district: 行政区，逗号分隔，如 海淀,朝阳
        area: 商圈，逗号分隔，如 西二旗,上地，国贸
        min_price: 最低月租金（元）
        max_price: 最高月租金（元）
        bedrooms: 卧室数，逗号分隔，如 1,2
        rental_type: 整租 或 合租
        decoration: 精装 或 简装
        orientation: 朝向，如 朝南、南北
        elevator: 是否有电梯:true/false
        min_area: 最小面积（平米）
        max_area: 最大面积（平米）
        property_type: 物业类型，如 住宅/公寓
        subway_line: 地铁线路，如 13号线
        max_subway_dist: 最大地铁距离（米），近地铁建议 800
        subway_station: 地铁站名，如 车公庄站
        utilities_type: 水电类型，如 民水民电
        available_from_before: 可入住日期上限,YYYY-MM-DD
        commute_to_xierqi_max: 到西二旗通勤时间上限（分钟）
        sort_by: 排序字段:price/area/subway
        sort_order: asc 或 desc
        normal_tags: 符合用户需求的tag,比如[%s]
        forbidden_tags: 要求不带有的tag,比如[%s],也包括normal_tag的取值
    """.format(positive_tags, negative_tags)


@mcp.tool()
async def get_houses_nearby(
    landmark_id: str,
    max_distance: float | None = None,
    listing_platform: str | None = None,
    normal_tags: list[str] | None = None,
    forbidden_tags: list[str] | None = None,
) -> str:
    
    params = _build_params(
        landmark_id=landmark_id, max_distance=max_distance,
        listing_platform=listing_platform, page=1, page_size=100,
    )
    tool_res = await _get("/api/houses/nearby", params, user_id=DEFAULT_USER_ID)
    all_houses = json.loads(tool_res)["data"]["items"]
    required_tag_set = set(normal_tags or [])
    forbidden_tag_set = set(forbidden_tags or [])
    scored_houses: list[tuple[int, dict]] = []

    for house in all_houses:
        house_tags = set(house.get("tags") or [])
        if forbidden_tag_set and forbidden_tag_set.intersection(house_tags):
            continue

        overlap_count = len(required_tag_set.intersection(house_tags))
        scored_houses.append((overlap_count, house))

    # Python sort is stable, so equal overlap_count keeps original all_houses order.
    scored_houses.sort(key=lambda x: x[0], reverse=True)
    ret_houses = [house for _, house in scored_houses[:5]]

    tool_res = json.dumps(ret_houses)
    return tool_res

get_houses_nearby.__doc__ = """以地标为圆心查询附近可租房源，返回直线距离、步行距离、步行时间。
    需先通过地标接口获得 landmark_id。
    Args:
        landmark_id: 地标 ID 或地标名称
        max_distance: 最大直线距离（米），默认 2000
        listing_platform: 挂牌平台：链家/安居客/58同城，不传默认安居客
        normal_tags: 符合用户需求的tag,比如[%s]
        forbidden_tags: 要求不带有的tag,比如[%s],也包括normal_tags的取值
    """.format(positive_tags, negative_tags)

# TODO:求解附近要求商场、公园是看tag还是看此接口？
@mcp.tool()
async def get_nearby_landmarks(
    community: str,
    type: str | None = None,
    max_distance_m: float | None = None,
) -> str:
    """查询某小区周边地标（商超/公园），按距离排序。用于回答「附近有没有商场/公园」。
    Args:
        community: 小区名，用于定位基准点
        type: 地标类型：shopping(商超) 或 park(公园)，不传则不过滤
        max_distance_m: 最大距离（米），默认 3000
    """
    params = _build_params(community=community, type=type, max_distance_m=max_distance_m)
    return await _get("/api/houses/nearby_landmarks", params, user_id=DEFAULT_USER_ID)


@mcp.tool()
async def get_house_stats() -> str:
    """获取房源统计信息（总套数、按状态/行政区/户型分布、价格区间等）。"""
    return await _get("/api/houses/stats", user_id=DEFAULT_USER_ID)


@mcp.tool()
async def rent_house(house_id: str, listing_platform: str) -> str:
    """租房 - 将该房源设为已租。必须调用此接口才算完成租房操作。
    Args:
        house_id: 房源 ID，如 HF_2001
        listing_platform: 必填，挂牌平台：链家/安居客/58同城
    """
    params = _build_params(listing_platform=listing_platform)
    tool_res = await _post(f"/api/houses/{house_id}/rent", params, user_id=DEFAULT_USER_ID)
    return "{}已租".format(house_id) + tool_res


@mcp.tool()
async def terminate_rental(house_id: str, listing_platform: str) -> str:
    """退租 - 将该房源恢复为可租。

    Args:
        house_id: 房源 ID，如 HF_2001
        listing_platform: 必填，挂牌平台：链家/安居客/58同城
    """
    params = _build_params(listing_platform=listing_platform)
    tool_res = await _post(f"/api/houses/{house_id}/terminate", params, user_id=DEFAULT_USER_ID)
    
    return "{}已退租".format(house_id) + tool_res


@mcp.tool()
async def take_offline(house_id: str, listing_platform: str) -> str:
    """下架 - 将该房源设为下架。

    Args:
        house_id: 房源 ID，如 HF_2001
        listing_platform: 必填，挂牌平台：链家/安居客/58同城
    """
    params = _build_params(listing_platform=listing_platform)
    tool_res = await _post(f"/api/houses/{house_id}/offline", params, user_id=DEFAULT_USER_ID)
    return "{}已下架".format(house_id) + tool_res


if __name__ == "__main__":
    mcp.run(transport="stdio")
