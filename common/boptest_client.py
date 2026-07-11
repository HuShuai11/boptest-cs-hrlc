"""Unified BOPTEST REST API client for all methods."""

import time
import numpy as np
import requests

BOPTEST_URL = "http://127.0.0.1"
DEFAULT_TESTCASE = "bestest_hydronic_heat_pump"
FORECAST_PRICE_POINT = "PriceElectricPowerHighlyDynamic"


def tout_to_c(t):
    t = float(t)
    return (t - 273.15) if t > 150.0 else t


def tout_to_c_array(arr):
    a = np.asarray(arr, dtype=float)
    return np.where(a > 150.0, a - 273.15, a)


class BoptestClient:
    """BOPTEST REST client — single source of truth for all methods."""

    def __init__(self, url=BOPTEST_URL, testcase=DEFAULT_TESTCASE, step_period=900):
        self.url = url
        self.testcase = testcase
        self.step_period = step_period
        self.testid = None
        self.last_res = {}

    # ------ lifecycle ------

    def start(self):
        # Connectivity check with retry (BOPTEST may be temporarily busy)
        for attempt in range(3):
            try:
                r = requests.get(f"{self.url}/testcases", timeout=5)
                if r.status_code == 200:
                    break
            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise ConnectionError(
                    f"BOPTEST not reachable at {self.url} after 3 attempts.\n"
                    f"Start it with: docker run --rm -p 80:80 --name boptest nrel/boptest:latest"
                )

        try:
            if self.testid:
                requests.put(f"{self.url}/stop/{self.testid}", timeout=120)
        except Exception:
            pass
        resp = requests.post(f"{self.url}/testcases/{self.testcase}/select", timeout=120)
        self.testid = resp.json()["testid"]
        return self.testid

    def initialize(self, start_time=0, warmup_period=12 * 3600):
        url = f"{self.url}/initialize/{self.testid}"
        payload = {"start_time": int(start_time), "warmup_period": int(warmup_period)}
        resp = requests.put(url, json=payload, timeout=120)
        self.last_res = resp.json()["payload"]
        return self.last_res

    def set_step(self, step_period=None):
        sp = step_period or self.step_period
        requests.put(f"{self.url}/step/{self.testid}", json={"step": int(sp)}, timeout=120)

    def set_scenario(self, scenario):
        requests.put(f"{self.url}/scenario/{self.testid}", json=scenario, timeout=120)

    def advance(self, actions_dict):
        resp = requests.post(f"{self.url}/advance/{self.testid}", json=actions_dict, timeout=120)
        self.last_res = resp.json()["payload"]
        return self.last_res

    # ------ data access ------

    def get_forecast(self, point_names, horizon=86400, interval=900):
        try:
            resp = requests.put(
                f"{self.url}/forecast/{self.testid}",
                json={"point_names": point_names, "horizon": int(horizon), "interval": int(interval)},
                timeout=120,
            )
            if resp.status_code != 200:
                return None
            return resp.json()["payload"]
        except Exception:
            return None

    def get_kpis(self):
        try:
            resp = requests.get(f"{self.url}/kpi/{self.testid}", timeout=120)
            return resp.json()["payload"]
        except Exception:
            # try without testid
            resp = requests.get(f"{self.url}/kpi", timeout=120)
            return resp.json().get("payload", resp.json())

    def stop(self):
        if self.testid:
            try:
                requests.put(f"{self.url}/stop/{self.testid}", timeout=120)
            except Exception:
                pass
