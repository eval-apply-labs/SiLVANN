// SiLVANN's browser companion: a fresh profile each start, in memory, nothing remembered.
user_pref("browser.shell.checkDefaultBrowser", false);
user_pref("browser.startup.homepage_override.mstone", "ignore");
user_pref("browser.aboutwelcome.enabled", false);
user_pref("datareporting.policy.dataSubmissionEnabled", false);
user_pref("datareporting.healthreport.uploadEnabled", false);
user_pref("toolkit.telemetry.enabled", false);
user_pref("app.update.enabled", false);
user_pref("browser.download.folderList", 2);
user_pref("browser.download.dir", "/tmp/downloads");
user_pref("browser.download.useDownloadDir", true);
user_pref("browser.download.always_ask_before_handling_new_types", false);
// a screen share joined without a camera or a microphone: fake devices, no prompt
user_pref("media.navigator.streams.fake", true);
user_pref("media.navigator.permission.disabled", true);
// no experiments, studies or new-tab content fetched in the background
user_pref("app.normandy.enabled", false);
user_pref("messaging-system.rsexperimentloader.enabled", false);
user_pref("browser.newtabpage.enabled", false);
user_pref("browser.startup.page", 0);
user_pref("extensions.pocket.enabled", false);
// Violentmonkey's pages at a fixed address, moz-extension://<this>/ — SiLVANN opens its editor to add a userscript
user_pref("extensions.webextensions.uuids", "{\"{aecec67f-0d10-4fa7-b7c7-609a2db280cf}\":\"5ac1a7a2-6f4b-4f0e-9a51-7a1d1e0c0f00\"}");
