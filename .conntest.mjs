const key = process.env.NVIDIA_NIM_API_KEY;
if(!key){ console.log("NO NIM KEY"); process.exit(0); }
const models = ["meta/llama-3.1-70b-instruct","meta/llama-3.3-70b-instruct","nvidia/llama-3.3-nemotron-super-49b-v1"];
for(const model of models){
  const t0=Date.now();
  try{
    const r=await fetch("https://integrate.api.nvidia.com/v1/chat/completions",{
      method:"POST",
      headers:{"Authorization":"Bearer "+key,"Content-Type":"application/json"},
      body:JSON.stringify({model,messages:[{role:"user",content:"Reply with exactly: heartbeat ok"}],max_tokens:20,temperature:0.2})
    });
    const j=await r.json();
    console.log(model,"→",r.status,Date.now()-t0+"ms",":",(j.choices?.[0]?.message?.content??JSON.stringify(j).slice(0,180)).replace(/\n/g,' '));
  }catch(e){ console.log(model,"ERR",e.message); }
}
