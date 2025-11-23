# ensemble.py
import os, time, json, requests, math

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY","").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY","").strip()
OLLAMA_HOST = os.getenv("OLLAMA_HOST","http://localhost:11434")  # for llama3 via Ollama

def ask_openai(prompt:str)->str|None:
    if not OPENAI_API_KEY: return None
    url="https://api.openai.com/v1/chat/completions"
    headers={"Authorization":f"Bearer {OPENAI_API_KEY}"}
    body={"model":"gpt-4o-mini","messages":[{"role":"user","content":prompt}],"temperature":0.1}
    r=requests.post(url,json=body,headers=headers,timeout=60)
    if r.status_code==200:
        return r.json()["choices"][0]["message"]["content"].strip()
    return None

def ask_deepseek(prompt:str)->str|None:
    if not DEEPSEEK_API_KEY: return None
    url="https://api.deepseek.com/chat/completions"
    headers={"Authorization":f"Bearer {DEEPSEEK_API_KEY}"}
    body={"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],"temperature":0.1}
    r=requests.post(url,json=body,headers=headers,timeout=60)
    if r.status_code==200:
        return r.json()["choices"][0]["message"]["content"].strip()
    return None

def ask_ollama(prompt:str, model="llama3")->str|None:
    url=f"{OLLAMA_HOST}/api/generate"
    try:
        r=requests.post(url,json={"model":model,"prompt":prompt,"stream":False},timeout=60)
        if r.ok: return r.json().get("response","").strip()
    except Exception:
        pass
    return None

def simple_score(ans:str, must_include:list[str]|None=None)->float:
    if not ans: return 0.0
    score = 0.0
    n = len(ans)
    score += min(n/800, 1.0) * 0.3            # not too short
    if any(h in ans for h in ["•","- ","1)","2)"]): score += 0.2  # structure
    if must_include:
        hits = sum(1 for k in must_include if k.lower() in ans.lower())
        score += 0.5 * (hits / max(1,len(must_include)))
    return round(score,3)

def adjudicate(prompt:str, must_include:list[str]|None=None)->dict:
    answers = {}
    answers["openai"]   = ask_openai(prompt)
    answers["deepseek"] = ask_deepseek(prompt)
    answers["llama"]    = ask_ollama(prompt)

    scored = []
    for k,v in answers.items():
        scored.append((k, simple_score(v, must_include), v))
    scored.sort(key=lambda x:x[1], reverse=True)

    return {
        "prompt": prompt,
        "winner": scored[0][0] if scored else None,
        "scores": [{"model":k,"score":s} for k,s,_ in scored],
        "answer": scored[0][2] if scored else None,
        "raw": answers
    }

if __name__=="__main__":
    q = "Summarize today’s top market drivers in 5 bullets. Include: CPI, Fed, earnings."
    out = adjudicate(q, must_include=["CPI","Fed","earnings"])
    print(json.dumps(out, indent=2))
